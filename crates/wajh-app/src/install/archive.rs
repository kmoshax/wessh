//! Pulling the shared libraries out of ONNX Runtime release archives (`.tgz` / `.zip`) and
//! NVIDIA's Python wheels (`.whl`, which are zip files). Everything else — headers, static
//! libraries, Python code — is skipped, and the libraries land flat in one directory.

use super::fetch::Cancelled;
use anyhow::{Context, Result, bail, ensure};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Directory names under which archives keep their shared libraries (TensorRT's wheels use
/// their own package directory instead of `lib/`).
const LIBRARY_DIRS: [&str; 3] = ["lib", "bin", "tensorrt_libs"];

/// If `member` (a `/`-separated archive path) is a shared library inside one of
/// [`LIBRARY_DIRS`], the file name to install it under.
#[must_use]
pub fn library_file_name(member: &str) -> Option<&str> {
    let mut parts = member.trim_end_matches('/').split('/');
    let name = parts.next_back()?;
    let in_library_dir = parts.any(|dir| LIBRARY_DIRS.contains(&dir));
    (in_library_dir && is_shared_library(name)).then_some(name)
}

/// `libfoo.so`, `libfoo.so.1.2`, `foo.dll`, `libfoo.1.dylib`.
#[must_use]
pub fn is_shared_library(name: &str) -> bool {
    let versioned_so = name
        .split_once(".so.")
        .is_some_and(|(_, version)| version.split('.').all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())));
    versioned_so || ["so", "dll", "dylib"].into_iter().any(|extension| has_extension(name, extension))
}

fn has_extension(name: &str, extension: &str) -> bool {
    name.rsplit_once('.').is_some_and(|(_, actual)| actual.eq_ignore_ascii_case(extension))
}

/// Extract every shared library in `archive` into `dest`; returns the installed file names.
/// Stops with [`Cancelled`] between files once `cancel` is set (every file is written whole).
pub fn extract_libraries(archive: &Path, dest: &Path, cancel: &AtomicBool) -> Result<Vec<String>> {
    std::fs::create_dir_all(dest)?;
    let name = archive.file_name().unwrap_or_default().to_string_lossy();
    let installed = if has_extension(&name, "tgz") || has_extension(&name, "gz") {
        extract_tar_gz(archive, dest, cancel)
    } else if has_extension(&name, "zip") || has_extension(&name, "whl") {
        extract_zip(archive, dest, cancel)
    } else {
        bail!("unknown archive format: {}", archive.display())
    }
    .map_err(|e| if e.is::<Cancelled>() { e } else { e.context(format!("failed to extract {}", archive.display())) })?;
    ensure!(!installed.is_empty(), "{} contains no shared libraries", archive.display());
    Ok(installed)
}

fn extract_tar_gz(archive: &Path, dest: &Path, cancel: &AtomicBool) -> Result<Vec<String>> {
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(File::open(archive)?));
    let mut installed = Vec::new();
    for entry in tar.entries()? {
        check(cancel)?;
        let mut entry = entry?;
        let member = entry.path()?.to_string_lossy().replace('\\', "/");
        let Some(name) = library_file_name(&member).map(str::to_owned) else { continue };
        let kind = entry.header().entry_type();
        if kind.is_symlink() {
            let target = entry.link_name()?.context("symlink without a target")?;
            let target = target.file_name().context("symlink to a directory")?;
            symlink(target, &dest.join(&name))?;
        } else if kind.is_file() {
            write_file(&mut entry, dest, &name)?;
        } else {
            continue;
        }
        installed.push(name);
    }
    Ok(installed)
}

fn extract_zip(archive: &Path, dest: &Path, cancel: &AtomicBool) -> Result<Vec<String>> {
    let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
    let mut installed = Vec::new();
    for index in 0..zip.len() {
        check(cancel)?;
        let mut file = zip.by_index(index)?;
        if !file.is_file() {
            continue;
        }
        let Some(name) = library_file_name(file.name()).map(str::to_owned) else { continue };
        write_file(&mut file, dest, &name)?;
        installed.push(name);
    }
    Ok(installed)
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) { Err(Cancelled.into()) } else { Ok(()) }
}

/// Write via a temporary name so an interrupted extraction never leaves a truncated library.
fn write_file(reader: &mut impl Read, dir: &Path, name: &str) -> Result<()> {
    let temporary = dir.join(format!("{name}.extracting"));
    io::copy(reader, &mut File::create(&temporary)?)?;
    std::fs::rename(&temporary, dir.join(name))?;
    Ok(())
}

#[cfg(unix)]
fn symlink(target: &std::ffi::OsStr, link: &Path) -> Result<()> {
    match std::fs::remove_file(link) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(not(unix))]
fn symlink(_target: &std::ffi::OsStr, link: &Path) -> Result<()> {
    bail!("{} is a symlink, which this platform's archives shouldn't contain", link.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_shared_libraries_from_library_dirs() {
        let keep = [
            ("onnxruntime-linux-x64-gpu_cuda13-1.28.2/lib/libonnxruntime.so.1.28.2", "libonnxruntime.so.1.28.2"),
            ("onnxruntime-linux-x64-gpu_cuda13-1.28.2/lib/libonnxruntime_providers_cuda.so", "libonnxruntime_providers_cuda.so"),
            ("onnxruntime-osx-arm64-1.28.2/lib/libonnxruntime.1.28.2.dylib", "libonnxruntime.1.28.2.dylib"),
            ("onnxruntime-win-x64-1.28.2/lib/onnxruntime.dll", "onnxruntime.dll"),
            ("nvidia/cu13/lib/libcudart.so.13", "libcudart.so.13"),
            ("nvidia/cu13/bin/x86_64/cudart64_13.dll", "cudart64_13.dll"),
            ("nvidia/cudnn/bin/cudnn64_9.dll", "cudnn64_9.dll"),
            ("tensorrt_libs/libnvinfer.so.10", "libnvinfer.so.10"),
        ];
        for (member, name) in keep {
            assert_eq!(library_file_name(member), Some(name), "{member}");
        }
    }

    #[test]
    fn skips_everything_else() {
        let skip = [
            "nvidia/cu13/lib/libcudart_static.a",
            "nvidia/cu13/lib/x64/cudart.lib",
            "nvidia/cu13/include/cuda.h",
            "onnxruntime-linux-x64-1.28.2/lib/pkgconfig/libonnxruntime.pc",
            "onnxruntime-linux-x64-1.28.2/lib/cmake/onnxruntime/onnxruntimeConfig.cmake",
            "onnxruntime-linux-x64-1.28.2/lib/",
            "onnxruntime-linux-x64-1.28.2/libonnxruntime.so",
            "nvidia_cuda_runtime-13.4.92.dist-info/RECORD",
            "tensorrt_libs/libnvinfer.so.10.debug",
            "tensorrt_libs/__init__.py",
        ];
        for member in skip {
            assert_eq!(library_file_name(member), None, "{member}");
        }
    }

    #[test]
    fn extracts_libraries_from_zip() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("wajh-archive-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wheel = dir.join("pkg.whl");
        let mut zip = zip::ZipWriter::new(File::create(&wheel).unwrap());
        for (name, body) in [("nvidia/cu13/lib/libcudart.so.13", "elf"), ("nvidia/cu13/include/cuda.h", "header")] {
            zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap();

        let dest = dir.join("out");
        assert_eq!(extract_libraries(&wheel, &dest, &AtomicBool::new(false)).unwrap(), ["libcudart.so.13"]);
        assert_eq!(std::fs::read_to_string(dest.join("libcudart.so.13")).unwrap(), "elf");
        assert!(!dest.join("cuda.h").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn extracts_libraries_and_symlinks_from_tar_gz() {
        let dir = std::env::temp_dir().join(format!("wajh-archive-tgz-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("ort.tgz");
        let gz = flate2::write::GzEncoder::new(File::create(&archive).unwrap(), flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        let mut header = tar::Header::new_gnu();
        header.set_size(3);
        header.set_mode(0o755);
        tar.append_data(&mut header, "ort/lib/libonnxruntime.so.1.28.2", &b"elf"[..]).unwrap();
        tar.append_data(&mut header, "ort/README.md", &b"doc"[..]).unwrap();
        let mut link = tar::Header::new_gnu();
        link.set_entry_type(tar::EntryType::Symlink);
        link.set_size(0);
        tar.append_link(&mut link, "ort/lib/libonnxruntime.so", "libonnxruntime.so.1.28.2").unwrap();
        tar.into_inner().unwrap().finish().unwrap();

        let dest = dir.join("out");
        let installed = extract_libraries(&archive, &dest, &AtomicBool::new(false)).unwrap();
        assert_eq!(installed, ["libonnxruntime.so.1.28.2", "libonnxruntime.so"]);
        assert_eq!(std::fs::read_to_string(dest.join("libonnxruntime.so")).unwrap(), "elf");
        assert!(!dest.join("README.md").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
