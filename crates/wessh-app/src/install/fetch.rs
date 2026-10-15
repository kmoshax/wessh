//! HTTP through the system `curl` (present on Linux, macOS and Windows 10+), which gives us
//! TLS, proxies, redirects, retries, resume and a progress bar without extra dependencies.

use anyhow::{Context, Result, bail, ensure};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const RETRIES: &str = "5";
const CONNECT_TIMEOUT_SECS: &str = "30";
const SHA256_HEX_LEN: usize = 64;
/// A transfer slower than this many bytes per second for [`STALL_SECS`] is treated as dropped
/// and resumed, so a stalled connection never hangs setup.
const STALL_BYTES_PER_SEC: &str = "1024";
const STALL_SECS: &str = "30";
/// Automatic resumes of a download that failed after making progress (flaky connections).
pub const AUTO_RESUMES: u32 = 10;
/// How often a running download reports its progress and checks for cancellation.
const POLL: Duration = Duration::from_millis(200);

/// A remote file to fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Download {
    pub url: String,
    /// Local file name (also names the cached archive).
    pub file_name: String,
    pub size: Option<u64>,
    /// Lower-case hex SHA-256, when the source publishes one.
    pub sha256: Option<String>,
}

fn curl() -> Command {
    let mut cmd = wessh_core::process::background_command("curl");
    cmd.args([
        "--fail",
        "--location",
        "--retry",
        RETRIES,
        "--connect-timeout",
        CONNECT_TIMEOUT_SECS,
        "--user-agent",
        concat!("wessh/", env!("CARGO_PKG_VERSION")),
    ]);
    cmd
}

fn run(cmd: &mut Command, url: &str) -> Result<Vec<u8>> {
    // Metadata is small: cap retries and the entire attempt so offline checks cannot hang.
    let output = cmd
        .args(["--max-time", "20", "--retry", "2", "--retry-max-time", "30"])
        .output()
        .context("could not run curl; install it and retry")?;
    if !output.status.success() {
        bail!("request to {url} failed: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(output.stdout)
}

/// Body of a GET request.
pub fn get_text(url: &str, accept: &str) -> Result<String> {
    let body = run(curl().args(["--silent", "--show-error", "--header", &format!("Accept: {accept}"), url]), url)?;
    String::from_utf8(body).with_context(|| format!("{url} returned invalid UTF-8"))
}

pub fn get_json<T: DeserializeOwned>(url: &str) -> Result<T> {
    let body = get_text(url, "application/json")?;
    serde_json::from_str(&body).with_context(|| format!("unexpected response from {url}"))
}

/// Bytes `start..end` of a remote file.
pub fn get_range(url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let body = run(curl().args(["--silent", "--show-error", "--range", &format!("{start}-{}", end - 1), url]), url)?;
    ensure!(body.len() as u64 == end - start, "{url} ignored the byte range (got {} bytes)", body.len());
    Ok(body)
}

/// Size and published checksum of a remote file, from a HEAD request.
pub fn head(url: &str) -> Result<(Option<u64>, Option<String>)> {
    let headers = run(curl().args(["--silent", "--show-error", "--head", url]), url)?;
    Ok(parse_head(&String::from_utf8_lossy(&headers)))
}

/// Size and SHA-256 from (possibly several, one per redirect) HTTP header blocks.
///
/// Hugging Face announces the LFS object in `x-linked-size` / `x-linked-etag` (a SHA-256) on the
/// redirect; otherwise the final response's `content-length` is used.
fn parse_head(headers: &str) -> (Option<u64>, Option<String>) {
    let (mut size, mut linked_size, mut sha256) = (None, None, None);
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => size = value.parse().ok(),
            "x-linked-size" => linked_size = value.parse().ok(),
            "x-linked-etag" => sha256 = normalize_sha256(value.trim_matches('"')),
            _ => {}
        }
    }
    (linked_size.or(size), sha256)
}

/// A lower-case SHA-256 hex digest, or `None` if `text` isn't one.
pub fn normalize_sha256(text: &str) -> Option<String> {
    let hex = text.strip_prefix("sha256:").unwrap_or(text);
    (hex.len() == SHA256_HEX_LEN && hex.bytes().all(|b| b.is_ascii_hexdigit())).then(|| hex.to_ascii_lowercase())
}

/// Download to `dest`, resuming a previous partial download, and verify the checksum.
/// An already complete, verified `dest` is kept as is.
///
/// `progress` is called with the bytes on disk (including those from earlier attempts) about
/// every [`POLL`]. A dropped connection that made progress is resumed up to [`AUTO_RESUMES`]
/// times; after that, or on any other failure, the partial file is kept so a later call resumes
/// it. Setting `cancel` stops curl and returns [`Cancelled`].
pub fn download(item: &Download, dest: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(Transfer)) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }
    if dest.exists() {
        match verify(item, dest, cancel, progress) {
            Ok(()) => return Ok(()),
            Err(error) if error.is::<Cancelled>() => return Err(error),
            Err(_) => {}
        }
    }
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = partial_path(dest);
    if item.size.is_some_and(|size| file_len(&partial) > size) {
        std::fs::remove_file(&partial)?; // longer than the real file: not a prefix of it
    }
    with_resumes(&item.file_name, &partial, |partial| {
        if item.size == Some(file_len(partial)) {
            return Ok(());
        }
        let mut cmd = curl();
        cmd.args(["--continue-at", "-", "--output"]).arg(partial).arg(&item.url).stdout(Stdio::null());
        transfer(cmd, &item.file_name, partial, cancel, &mut |bytes| progress(Transfer::Bytes(bytes)))
    })?;
    if let Err(e) = verify(item, &partial, cancel, progress) {
        if e.is::<Cancelled>() {
            return Err(e);
        }
        std::fs::remove_file(&partial)?;
        return Err(e.context("the downloaded file was damaged and has been discarded; try again"));
    }
    // Windows cannot rename over an existing file. Keep it until its replacement is verified.
    if dest.exists() {
        std::fs::remove_file(dest)?;
    }
    std::fs::rename(&partial, dest)?;
    Ok(())
}

/// What [`download`] is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    /// This many bytes of the file are on disk.
    Bytes(u64),
    /// Checking the finished file's size and checksum.
    Verifying,
}

/// The download was cancelled; its partial file is kept for the next attempt.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Bytes of `path` on disk, 0 if it doesn't exist.
#[must_use]
pub fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

/// Bytes `start..end` of `url` into `partial`, resuming what it already holds, polled like
/// [`download`]; `progress` gets the bytes of the range on disk.
pub fn download_range(url: &str, start: u64, end: u64, partial: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(u64)) -> Result<()> {
    let len = end - start;
    if file_len(partial) > len {
        std::fs::remove_file(partial)?;
    }
    let name = partial.file_name().unwrap_or_default().to_string_lossy().into_owned();
    with_resumes(&name, partial, |partial| {
        let have = file_len(partial);
        if have == len {
            return Ok(());
        }
        let output = File::options().create(true).append(true).open(partial)?;
        let mut cmd = curl();
        cmd.args(["--range", &format!("{}-{}", start + have, end - 1), url]).stdout(output);
        transfer(cmd, &name, partial, cancel, progress)
    })?;
    ensure!(file_len(partial) == len, "{name}: the server sent {} of {len} bytes", file_len(partial));
    Ok(())
}

/// Run `attempt` (which returns early when the file is already complete) until it succeeds. An
/// attempt that fails after making progress (a dropped connection) is retried up to
/// [`AUTO_RESUMES`] times.
fn with_resumes(name: &str, partial: &Path, mut attempt: impl FnMut(&Path) -> Result<()>) -> Result<()> {
    let mut resumes = 0;
    loop {
        let before = file_len(partial);
        match attempt(partial) {
            Ok(()) => return Ok(()),
            Err(e) if e.is::<Cancelled>() => return Err(e),
            Err(e) if file_len(partial) > before && resumes < AUTO_RESUMES => {
                resumes += 1;
                eprintln!("{name}: {e:#}; resuming ({resumes}/{AUTO_RESUMES})");
            }
            Err(e) => return Err(e),
        }
    }
}

/// Run a curl download `cmd` that writes to `partial`, polled for progress and cancellation.
fn transfer(mut cmd: Command, name: &str, partial: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(u64)) -> Result<()> {
    let mut child = cmd
        .args(["--silent", "--show-error", "--speed-limit", STALL_BYTES_PER_SEC, "--speed-time", STALL_SECS])
        .stderr(Stdio::piped())
        .spawn()
        .context("could not run curl; install it and retry")?;
    let status = loop {
        progress(file_len(partial));
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Cancelled.into());
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        std::thread::sleep(POLL);
    };
    progress(file_len(partial));
    if status.success() {
        return Ok(());
    }
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    let reason = stderr.trim().trim_start_matches("curl: ").to_owned();
    let reason = if reason.is_empty() { status.to_string() } else { reason };
    bail!("download of {name} failed: {reason}")
}

/// Where the unfinished download of `dest` is kept.
#[must_use]
pub fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn verify(item: &Download, path: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(Transfer)) -> Result<()> {
    let len = file_len(path);
    if let Some(size) = item.size {
        ensure!(len == size, "{} is {len} bytes, expected {size}", path.display());
    }
    if let Some(expected) = &item.sha256 {
        progress(Transfer::Verifying);
        let actual = sha256_file(path, cancel)?;
        ensure!(&actual == expected, "checksum mismatch for {}: expected {expected}, got {actual}", item.file_name);
    }
    Ok(())
}

/// Lower-case hex SHA-256 of a file; checks `cancel` between chunks (large wheels take a while).
pub fn sha256_file(path: &Path, cancel: &AtomicBool) -> Result<String> {
    const CHUNK: usize = 1 << 20;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; CHUNK];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Cancelled.into());
        }
        match file.read(&mut buffer)? {
            0 => break,
            n => hasher.update(&buffer[..n]),
        }
    }
    Ok(hasher.finalize().iter().fold(String::with_capacity(SHA256_HEX_LEN), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada";

    #[test]
    fn head_prefers_linked_lfs_metadata() {
        let headers = format!(
            "HTTP/2 302\r\ncontent-length: 1032\r\nx-linked-size: 3295067\r\nx-linked-etag: \"{HASH}\"\r\n\r\nHTTP/2 200\r\ncontent-length: 3295067\r\n"
        );
        assert_eq!(parse_head(&headers), (Some(3_295_067), Some(HASH.to_owned())));
    }

    #[test]
    fn head_uses_last_content_length() {
        let headers = "HTTP/2 302\r\ncontent-length: 0\r\n\r\nHTTP/2 200\r\nContent-Length: 9128991\r\netag: \"0x8DE\"\r\n";
        assert_eq!(parse_head(headers), (Some(9_128_991), None));
    }

    #[test]
    fn sha256_normalization() {
        assert_eq!(normalize_sha256(&format!("sha256:{}", HASH.to_uppercase())).as_deref(), Some(HASH));
        assert_eq!(normalize_sha256("d1a5fa81"), None);
        assert_eq!(normalize_sha256(&"z".repeat(SHA256_HEX_LEN)), None);
    }

    #[test]
    fn repairs_corrupt_cached_file_from_verified_partial_without_network() {
        let dir = std::env::temp_dir().join(format!("wessh-fetch-repair-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model");
        std::fs::write(&file, "bad").unwrap();
        std::fs::write(partial_path(&file), "abc").unwrap();
        let item = Download {
            url: "http://127.0.0.1:0/unreachable".into(),
            file_name: "model".into(),
            size: Some(3),
            sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into()),
        };
        download(&item, &file, &AtomicBool::new(false), &mut |_| {}).unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"abc");
        assert!(!partial_path(&file).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancellation_preserves_existing_and_partial_files() {
        let dir = std::env::temp_dir().join(format!("wessh-fetch-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model");
        std::fs::write(&file, "cached").unwrap();
        std::fs::write(partial_path(&file), "partial").unwrap();
        let item = Download { url: String::new(), file_name: "model".into(), size: Some(6), sha256: None };
        assert!(
            download(&item, &file, &AtomicBool::new(true), &mut |_| panic!("cancelled download reported progress"))
                .unwrap_err()
                .is::<Cancelled>()
        );
        assert_eq!(std::fs::read(&file).unwrap(), b"cached");
        assert_eq!(std::fs::read(partial_path(&file)).unwrap(), b"partial");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hashes_files() {
        let dir = std::env::temp_dir().join(format!("wessh-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc");
        std::fs::write(&file, "abc").unwrap();
        assert_eq!(
            sha256_file(&file, &AtomicBool::new(false)).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
