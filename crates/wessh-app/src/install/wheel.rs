//! Installing only some files of a remote wheel. Wheels are zip files, whose central directory
//! (at the end) says where each member's compressed bytes are, so the members we need are
//! fetched with HTTP range requests and the rest is never downloaded: the TensorRT for RTX wheel
//! carries a 49 MB plugin library nothing loads, cuDNN's a 100 MB one for RNNs and attention.
//!
//! Without the whole file its published SHA-256 can't be checked; instead every member is checked
//! against the SHA-256 the wheel's own `RECORD` lists for it (and its zip CRC-32 and size).
//! Each member's range is downloaded resumably into its own partial file, so a pause or a dropped
//! connection loses nothing.

use super::fetch::{self, Cancelled};
use anyhow::{Context, Result, bail, ensure};
use flate2::read::DeflateDecoder;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// End of central directory record: signature and fixed size.
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const EOCD_LEN: usize = 22;
/// The zip comment after it is at most this long.
const MAX_COMMENT: usize = u16::MAX as usize;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const CENTRAL_LEN: usize = 46;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const LOCAL_LEN: usize = 30;
/// Zip64 marks sizes and offsets it moved elsewhere with all-ones; wheels never need it.
const ZIP64_MARKER: u32 = u32::MAX;
const STORED: u16 = 0;
const DEFLATED: u16 = 8;

/// One file inside a remote zip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    method: u16,
    crc32: u32,
    compressed: u64,
    pub size: u64,
    /// Byte range from its local header to the next member (or the central directory).
    start: u64,
    end: u64,
}

impl Member {
    /// Bytes to download for it.
    #[must_use]
    pub const fn download_len(&self) -> u64 {
        self.end - self.start
    }
}

/// The members of the zip at `url` (`size` bytes long), from its central directory.
pub fn members(url: &str, size: u64) -> Result<Vec<Member>> {
    let tail_len = size.min((EOCD_LEN + MAX_COMMENT) as u64);
    let tail = fetch::get_range(url, size - tail_len, size)?;
    let eocd = (0..=tail.len().saturating_sub(EOCD_LEN))
        .rev()
        .find(|&i| u32_at(&tail, i) == EOCD_SIGNATURE)
        .context("not a zip file: no end of central directory")?;
    let (cd_len, cd_offset) = (u64::from(u32_at(&tail, eocd + 12)), u64::from(u32_at(&tail, eocd + 16)));
    ensure!(cd_offset != u64::from(ZIP64_MARKER) && cd_offset + cd_len <= size, "unsupported zip (zip64)");
    let tail_start = size - tail_len;
    let directory = if cd_offset >= tail_start {
        let from = usize::try_from(cd_offset - tail_start)?;
        tail[from..from + usize::try_from(cd_len)?].to_vec()
    } else {
        fetch::get_range(url, cd_offset, cd_offset + cd_len)?
    };
    parse_directory(&directory, cd_offset)
}

/// Central directory entries, each ending where the next member (by offset) begins.
fn parse_directory(directory: &[u8], cd_offset: u64) -> Result<Vec<Member>> {
    let mut members = Vec::new();
    let mut at = 0;
    while at + CENTRAL_LEN <= directory.len() && u32_at(directory, at) == CENTRAL_SIGNATURE {
        let (name_len, extra_len, comment_len) =
            (usize::from(u16_at(directory, at + 28)), usize::from(u16_at(directory, at + 30)), usize::from(u16_at(directory, at + 32)));
        let name = directory.get(at + CENTRAL_LEN..at + CENTRAL_LEN + name_len).context("truncated zip directory")?;
        let (compressed, size, start) = (u32_at(directory, at + 20), u32_at(directory, at + 24), u32_at(directory, at + 42));
        ensure!(![compressed, size, start].contains(&ZIP64_MARKER), "unsupported zip (zip64)");
        members.push(Member {
            name: String::from_utf8_lossy(name).into_owned(),
            method: u16_at(directory, at + 10),
            crc32: u32_at(directory, at + 16),
            compressed: u64::from(compressed),
            size: u64::from(size),
            start: u64::from(start),
            end: 0,
        });
        at += CENTRAL_LEN + name_len + extra_len + comment_len;
    }
    ensure!(!members.is_empty(), "empty zip directory");
    let mut starts: Vec<u64> = members.iter().map(|m| m.start).chain([cd_offset]).collect();
    starts.sort_unstable();
    for member in &mut members {
        member.end = starts[starts.partition_point(|&s| s <= member.start)];
    }
    Ok(members)
}

/// What to fetch from a wheel: `files`, and its `RECORD` to check them against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub url: String,
    pub record: Member,
    /// Each member and the file name to install it under.
    pub files: Vec<(Member, String)>,
}

impl Selection {
    /// `pick` names the file each wanted member is installed as.
    pub fn new(url: &str, size: u64, pick: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let members = members(url, size)?;
        let record = members.iter().find(|m| m.name.ends_with(".dist-info/RECORD")).context("wheel has no RECORD")?.clone();
        let files: Vec<_> = members.into_iter().filter_map(|m| pick(&m.name).map(|name| (m, name))).collect();
        ensure!(!files.is_empty(), "the wheel has none of the files needed");
        Ok(Self { url: url.to_owned(), record, files })
    }

    /// Bytes to download.
    #[must_use]
    pub fn download_len(&self) -> u64 {
        self.record.download_len() + self.files.iter().map(|(m, _)| m.download_len()).sum::<u64>()
    }

    /// Bytes downloaded by earlier, interrupted runs.
    #[must_use]
    pub fn partial_len(&self, parts: &Path) -> u64 {
        std::iter::once(&self.record).chain(self.files.iter().map(|(m, _)| m)).map(|m| fetch::file_len(&part_path(parts, m))).sum()
    }

    /// Download the members into `dir` (their ranges kept in `parts` until installed);
    /// `progress` gets the bytes downloaded so far.
    pub fn install(&self, dir: &Path, parts: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(u64)) -> Result<Vec<PathBuf>> {
        std::fs::create_dir_all(dir)?;
        std::fs::create_dir_all(parts)?;
        let mut done = 0;
        let record = String::from_utf8(self.fetch_member(&self.record, parts, cancel, &mut done, progress)?)?;
        let mut installed = Vec::new();
        for (member, name) in &self.files {
            let expected = record_sha256(&record, &member.name).with_context(|| format!("RECORD lists no hash for {}", member.name))?;
            let path = dir.join(name);
            if fetch::file_len(&path) == member.size && file_sha256(&path)? == expected {
                done += member.download_len();
                progress(done);
            } else {
                let part = self.download(member, parts, cancel, &mut done, progress)?;
                extract(&part, member, &path, &expected).with_context(|| format!("{} is damaged; try again", member.name))?;
                std::fs::remove_file(part)?;
            }
            installed.push(path);
        }
        std::fs::remove_file(part_path(parts, &self.record))?;
        let _ = std::fs::remove_dir(parts);
        Ok(installed)
    }

    /// A small member's contents.
    fn fetch_member(
        &self,
        member: &Member,
        parts: &Path,
        cancel: &AtomicBool,
        done: &mut u64,
        progress: &mut dyn FnMut(u64),
    ) -> Result<Vec<u8>> {
        let part = self.download(member, parts, cancel, done, progress)?;
        let mut contents = Vec::new();
        member_reader(&part, member)?.read_to_end(&mut contents)?;
        ensure!(contents.len() as u64 == member.size, "{} has the wrong size", member.name);
        Ok(contents)
    }

    /// `member`'s range, downloaded (or resumed) into its partial file.
    fn download(
        &self,
        member: &Member,
        parts: &Path,
        cancel: &AtomicBool,
        done: &mut u64,
        progress: &mut dyn FnMut(u64),
    ) -> Result<PathBuf> {
        if cancel.load(Ordering::Relaxed) {
            return Err(Cancelled.into());
        }
        let part = part_path(parts, member);
        let before = *done;
        fetch::download_range(&self.url, member.start, member.end, &part, cancel, &mut |bytes| progress(before + bytes))?;
        *done = before + member.download_len();
        Ok(part)
    }
}

/// Where a member's downloaded range is kept until it is installed.
fn part_path(parts: &Path, member: &Member) -> PathBuf {
    let name: String = member.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '_' }).collect();
    parts.join(format!("{name}.part"))
}

/// The decompressed contents of `member`, from its downloaded range.
fn member_reader(part: &Path, member: &Member) -> Result<Box<dyn Read>> {
    let mut file = File::open(part)?;
    let mut header = [0; LOCAL_LEN];
    file.read_exact(&mut header)?;
    ensure!(u32_at(&header, 0) == LOCAL_SIGNATURE, "no local header for {}", member.name);
    let skip = usize::from(u16_at(&header, 26)) + usize::from(u16_at(&header, 28));
    file.seek(SeekFrom::Start((LOCAL_LEN + skip) as u64))?;
    let data = file.take(member.compressed);
    Ok(match member.method {
        STORED => Box::new(data),
        DEFLATED => Box::new(DeflateDecoder::new(data)),
        method => bail!("{} uses unsupported compression {method}", member.name),
    })
}

/// Decompress `member` to `dest` through a temporary name, checking its size, CRC-32 and SHA-256.
fn extract(part: &Path, member: &Member, dest: &Path, sha256: &str) -> Result<()> {
    let temporary = dest.with_extension("extracting");
    let mut reader = member_reader(part, member)?;
    let (mut output, mut hasher, mut crc) = (File::create(&temporary)?, Sha256::new(), flate2::Crc::new());
    let mut buffer = vec![0; 1 << 20];
    let mut written = 0u64;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        output.write_all(&buffer[..n])?;
        hasher.update(&buffer[..n]);
        crc.update(&buffer[..n]);
        written += n as u64;
    }
    output.sync_all()?;
    drop(output);
    let checked = (|| {
        ensure!(written == member.size, "{written} bytes, expected {}", member.size);
        ensure!(crc.sum() == member.crc32, "CRC-32 mismatch");
        ensure!(base64url(&hasher.finalize()) == sha256, "SHA-256 differs from the wheel's RECORD");
        Ok(())
    })();
    if let Err(e) = checked {
        let _ = std::fs::remove_file(&temporary);
        let _ = std::fs::remove_file(part);
        return Err(e);
    }
    std::fs::rename(&temporary, dest)?;
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let (mut file, mut hasher, mut buffer) = (File::open(path)?, Sha256::new(), vec![0; 1 << 20]);
    loop {
        match file.read(&mut buffer)? {
            0 => return Ok(base64url(&hasher.finalize())),
            n => hasher.update(&buffer[..n]),
        }
    }
}

/// The `sha256=` digest `RECORD` lists for `name` (unpadded URL-safe base64, as listed).
fn record_sha256(record: &str, name: &str) -> Option<String> {
    record.lines().find_map(|line| {
        let mut fields = line.rsplitn(3, ',');
        let (_size, hash, path) = (fields.next()?, fields.next()?, fields.next()?);
        (path == name).then(|| hash.strip_prefix("sha256=").map(str::to_owned)).flatten()
    })
}

/// Unpadded URL-safe base64 (RFC 4648 §5), the encoding of wheel `RECORD` hashes.
fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(char::from(ALPHABET[(n >> (18 - 6 * i) & 63) as usize]));
        }
    }
    out
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    bytes.get(at..at + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    bytes.get(at..at + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A small wheel-like zip, written by the `zip` crate.
    fn wheel() -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let deflated = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("pkg/libbig.so", deflated).unwrap();
        zip.write_all(&vec![7; 10_000]).unwrap();
        zip.start_file("pkg/libsmall.so", zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored))
            .unwrap();
        zip.write_all(b"small").unwrap();
        zip.start_file("pkg-1.0.dist-info/RECORD", deflated).unwrap();
        zip.write_all(b"pkg/libsmall.so,sha256=x,5\n").unwrap();
        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn reads_members_and_their_ranges() {
        let bytes = wheel();
        let eocd = bytes.len() - EOCD_LEN;
        let (len, offset) = (u32_at(&bytes, eocd + 12) as usize, u32_at(&bytes, eocd + 16) as usize);
        let members = parse_directory(&bytes[offset..offset + len], offset as u64).unwrap();
        let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["pkg/libbig.so", "pkg/libsmall.so", "pkg-1.0.dist-info/RECORD"]);
        assert_eq!(members[0].end, members[1].start);
        assert_eq!(members[2].end, offset as u64);
        assert!(members[0].compressed < 1_000, "deflated: {}", members[0].compressed);

        // Each member decompresses from its own range alone.
        let dir = std::env::temp_dir().join(format!("wessh-wheel-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for member in &members[..2] {
            let part = dir.join("part");
            std::fs::write(&part, &bytes[member.start as usize..member.end as usize]).unwrap();
            let mut contents = Vec::new();
            member_reader(&part, member).unwrap().read_to_end(&mut contents).unwrap();
            assert_eq!(contents.len() as u64, member.size);
            let sha = base64url(&Sha256::digest(&contents));
            extract(&part, member, &dir.join("out"), &sha).unwrap();
            assert_eq!(std::fs::read(dir.join("out")).unwrap(), contents);
            std::fs::write(&part, &bytes[member.start as usize..member.end as usize]).unwrap();
            assert!(extract(&part, member, &dir.join("out2"), "wrong").is_err());
            assert!(!dir.join("out2").exists());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn record_hashes_and_base64url() {
        let record = "a/b.so,sha256=AbC-_x,12\nc,d.so,sha256=zzz,3\npkg.dist-info/RECORD,,\n";
        assert_eq!(record_sha256(record, "a/b.so").as_deref(), Some("AbC-_x"));
        assert_eq!(record_sha256(record, "c,d.so").as_deref(), Some("zzz"));
        assert_eq!(record_sha256(record, "pkg.dist-info/RECORD"), None);
        assert_eq!(base64url(b""), "");
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
    }
}
