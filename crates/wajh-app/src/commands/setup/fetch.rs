//! HTTP through the system `curl` (present on Linux, macOS and Windows 10+), which gives us
//! TLS, proxies, redirects, retries, resume and a progress bar without extra dependencies.

use anyhow::{Context, Result, bail, ensure};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const RETRIES: &str = "5";
const CONNECT_TIMEOUT_SECS: &str = "30";
const SHA256_HEX_LEN: usize = 64;

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
    let mut cmd = Command::new("curl");
    cmd.args([
        "--fail",
        "--location",
        "--retry",
        RETRIES,
        "--connect-timeout",
        CONNECT_TIMEOUT_SECS,
        "--user-agent",
        concat!("wajh/", env!("CARGO_PKG_VERSION")),
    ]);
    cmd
}

fn run(cmd: &mut Command, url: &str) -> Result<Vec<u8>> {
    let output = cmd.output().context("could not run curl; install it and retry")?;
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
pub fn download(item: &Download, dest: &Path) -> Result<()> {
    if dest.exists() && verify(item, dest).is_ok() {
        return Ok(());
    }
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = partial_path(dest);
    let already = std::fs::metadata(&partial).map_or(0, |m| m.len());
    if item.size != Some(already) {
        let status = curl()
            .args(["--continue-at", "-", "--progress-bar", "--output"])
            .arg(&partial)
            .arg(&item.url)
            .status()
            .context("could not run curl; install it and retry")?;
        ensure!(status.success(), "download of {} failed ({status}); run the command again to resume", item.url);
    }
    if let Err(e) = verify(item, &partial) {
        std::fs::remove_file(&partial)?;
        return Err(e.context("the partial download was discarded; run the command again"));
    }
    std::fs::rename(&partial, dest)?;
    Ok(())
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn verify(item: &Download, path: &Path) -> Result<()> {
    let len = std::fs::metadata(path)?.len();
    if let Some(size) = item.size {
        ensure!(len == size, "{} is {len} bytes, expected {size}", path.display());
    }
    if let Some(expected) = &item.sha256 {
        let actual = sha256_file(path)?;
        ensure!(&actual == expected, "checksum mismatch for {}: expected {expected}, got {actual}", item.file_name);
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    const CHUNK: usize = 1 << 20;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; CHUNK];
    loop {
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
    fn hashes_files() {
        let dir = std::env::temp_dir().join(format!("wajh-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc");
        std::fs::write(&file, "abc").unwrap();
        assert_eq!(sha256_file(&file).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
