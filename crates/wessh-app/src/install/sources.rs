//! Where each component comes from: ONNX Runtime GitHub releases, NVIDIA's CUDA wheels on
//! PyPI (TensorRT on NVIDIA's own index), the models on Hugging Face, and a static ffmpeg from
//! the `imageio-ffmpeg` wheels on PyPI.

use super::fetch::{self, Download};
use super::platform::{Arch, Flavour, Os, Target};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use wessh_core::assets::Asset;

/// ONNX Runtime release that `ort` 2.0.0-rc.13 (`api-28`) is built and tested against.
pub const ORT_VERSION: &str = "1.28.2";
const ORT_REPO: &str = "microsoft/onnxruntime";
const PYPI: &str = "https://pypi.org/pypi";
const NVIDIA_INDEX: &str = "https://pypi.nvidia.com";

/// File name of the ONNX Runtime release archive for a platform and flavour.
pub fn ort_archive_name(target: Target, flavour: Flavour, version: &str) -> Result<String> {
    let platform = match (target.os, target.arch, flavour) {
        (Os::Linux, Arch::X86_64, Flavour::Cpu) => "linux-x64",
        (Os::Linux, Arch::X86_64, Flavour::Cuda12) => "linux-x64-gpu_cuda12",
        (Os::Linux, Arch::X86_64, Flavour::Cuda13) => "linux-x64-gpu_cuda13",
        (Os::Linux, Arch::Aarch64, Flavour::Cpu) => "linux-aarch64",
        (Os::Windows, Arch::X86_64, Flavour::Cpu) => "win-x64",
        (Os::Windows, Arch::X86_64, Flavour::Cuda12) => "win-x64-gpu_cuda12",
        (Os::Windows, Arch::X86_64, Flavour::Cuda13) => "win-x64-gpu_cuda13",
        (Os::Windows, Arch::Aarch64, Flavour::Cpu) => "win-arm64",
        (Os::Mac, Arch::Aarch64, Flavour::Cpu | Flavour::Coreml) => "osx-arm64",
        (Os::Mac, Arch::X86_64, _) => bail!(
            "Microsoft no longer publishes ONNX Runtime for Intel Macs; run `brew install onnxruntime` and start wessh with \
             --ort \"$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib\""
        ),
        (os, arch, flavour) => bail!("there is no {flavour} build of ONNX Runtime for {os:?} {arch:?}; try --flavour cpu"),
    };
    let extension = if target.os == Os::Windows { "zip" } else { "tgz" };
    Ok(format!("onnxruntime-{platform}-{version}.{extension}"))
}

#[derive(Deserialize)]
struct Release {
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
    size: u64,
    browser_download_url: String,
    /// `sha256:<hex>`, published for releases since mid-2025.
    digest: Option<String>,
}

/// The ONNX Runtime archive with its size and checksum from the GitHub releases API. If the API
/// is unreachable (e.g. rate limited) the well-known download URL is used without a checksum.
pub fn ort_release(target: Target, flavour: Flavour, version: &str) -> Result<Download> {
    let name = ort_archive_name(target, flavour, version)?;
    let api = format!("https://api.github.com/repos/{ORT_REPO}/releases/tags/v{version}");
    match fetch::get_json::<Release>(&api) {
        Ok(release) => {
            let Some(asset) = release.assets.into_iter().find(|a| a.name == name) else {
                bail!("ONNX Runtime v{version} has no asset named {name}");
            };
            Ok(Download {
                url: asset.browser_download_url,
                file_name: asset.name,
                size: Some(asset.size),
                sha256: asset.digest.as_deref().and_then(fetch::normalize_sha256),
            })
        }
        Err(e) => {
            eprintln!("warning: GitHub API unavailable ({e:#}); downloading without checksum verification");
            let url = format!("https://github.com/{ORT_REPO}/releases/download/v{version}/{name}");
            let (size, sha256) = fetch::head(&url)?;
            Ok(Download { url, file_name: name, size, sha256 })
        }
    }
}

/// Package index hosting a wheel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Index {
    PyPi,
    /// `pypi.nvidia.com`; PyPI only has placeholder sdists for TensorRT.
    Nvidia,
}

/// An NVIDIA redistributable shipped as a Python wheel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wheel {
    pub package: &'static str,
    /// Name people know the library by.
    pub label: &'static str,
    /// Only versions starting with this are compatible (pins the CUDA generation).
    pub versions: &'static str,
    /// File-name prefix of the main library, per `[unix, windows]`; proves it is installed.
    library: [&'static str; 2],
    index: Index,
}

const fn wheel(package: &'static str, label: &'static str, versions: &'static str, unix: &'static str, windows: &'static str) -> Wheel {
    Wheel { package, label, versions, library: [unix, windows], index: Index::PyPi }
}

/// The CUDA 13 libraries ONNX Runtime's CUDA provider and cuDNN load.
const CUDA13: [Wheel; 7] = [
    wheel("nvidia-cuda-runtime", "CUDA runtime", "13.", "libcudart.so", "cudart64_"),
    wheel("nvidia-cublas", "cuBLAS", "13.", "libcublasLt.so", "cublasLt64_"),
    wheel("nvidia-cufft", "cuFFT", "12.", "libcufft.so", "cufft64_"),
    wheel("nvidia-curand", "cuRAND", "10.", "libcurand.so", "curand64_"),
    wheel("nvidia-cudnn-cu13", "cuDNN", "9.", "libcudnn.so", "cudnn64_"),
    wheel("nvidia-nvjitlink", "nvJitLink", "13.", "libnvJitLink.so", "nvJitLink_"),
    wheel("nvidia-cuda-nvrtc", "NVRTC", "13.", "libnvrtc.so", "nvrtc64_"),
];

const CUDA12: [Wheel; 7] = [
    wheel("nvidia-cuda-runtime-cu12", "CUDA runtime", "12.", "libcudart.so", "cudart64_"),
    wheel("nvidia-cublas-cu12", "cuBLAS", "12.", "libcublasLt.so", "cublasLt64_"),
    wheel("nvidia-cufft-cu12", "cuFFT", "11.", "libcufft.so", "cufft64_"),
    wheel("nvidia-curand-cu12", "cuRAND", "10.", "libcurand.so", "curand64_"),
    wheel("nvidia-cudnn-cu12", "cuDNN", "9.", "libcudnn.so", "cudnn64_"),
    wheel("nvidia-nvjitlink-cu12", "nvJitLink", "12.", "libnvJitLink.so", "nvJitLink_"),
    wheel("nvidia-cuda-nvrtc-cu12", "NVRTC", "12.", "libnvrtc.so", "nvrtc64_"),
];

/// ONNX Runtime 1.28's TensorRT provider is built against TensorRT 10.
const fn tensorrt(package: &'static str) -> Wheel {
    Wheel { package, label: "TensorRT", versions: "10.", library: ["libnvinfer.so", "nvinfer_"], index: Index::Nvidia }
}

/// Wheels needed for a CUDA major version, optionally with TensorRT.
#[must_use]
pub fn cuda_wheels(cuda_major: u32, with_tensorrt: bool) -> Vec<Wheel> {
    let (base, trt) = if cuda_major >= 13 { (&CUDA13, tensorrt("tensorrt-cu13-libs")) } else { (&CUDA12, tensorrt("tensorrt-cu12-libs")) };
    base.iter().copied().chain(with_tensorrt.then_some(trt)).collect()
}

impl Wheel {
    /// TensorRT, the optional (and by far the largest) download.
    #[must_use]
    pub fn is_tensorrt(&self) -> bool {
        self.index == Index::Nvidia
    }

    /// Whether this wheel's main library is already in `dir`.
    #[must_use]
    pub fn is_installed(&self, dir: &Path, os: Os) -> bool {
        let prefix = self.library[usize::from(os == Os::Windows)];
        std::fs::read_dir(dir).is_ok_and(|entries| entries.flatten().any(|e| e.file_name().to_string_lossy().starts_with(prefix)))
    }

    /// Newest compatible wheel for `target`: its version and download.
    pub fn resolve(&self, target: Target) -> Result<(String, Download)> {
        let candidates = match self.index {
            Index::PyPi => pypi_candidates(self.package)?,
            Index::Nvidia => {
                let url = format!("{NVIDIA_INDEX}/{}/", self.package);
                parse_simple_index(&fetch::get_text(&url, "text/html")?, &url)
            }
        };
        let best = pick(candidates, self.versions, target)
            .with_context(|| format!("no {} {}x wheel for {:?} {:?}", self.package, self.versions, target.os, target.arch))?;
        let (size, sha256) = match best.size {
            Some(size) => (Some(size), best.sha256),
            None => (fetch::head(&best.url)?.0, best.sha256),
        };
        Ok((best.version, Download { url: best.url, file_name: best.filename, size, sha256 }))
    }
}

/// One downloadable wheel file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Candidate {
    version: String,
    filename: String,
    url: String,
    size: Option<u64>,
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct PypiProject {
    releases: HashMap<String, Vec<PypiFile>>,
}

#[derive(Deserialize)]
struct PypiFile {
    filename: String,
    url: String,
    size: u64,
    digests: PypiDigests,
    #[serde(default)]
    yanked: bool,
}

#[derive(Deserialize)]
struct PypiDigests {
    sha256: String,
}

fn pypi_candidates(package: &str) -> Result<Vec<Candidate>> {
    let project: PypiProject = fetch::get_json(&format!("{PYPI}/{package}/json"))?;
    Ok(project
        .releases
        .into_iter()
        .flat_map(|(version, files)| {
            files.into_iter().filter(|f| !f.yanked).map(move |f| Candidate {
                version: version.clone(),
                filename: f.filename,
                url: f.url,
                size: Some(f.size),
                sha256: fetch::normalize_sha256(&f.digests.sha256),
            })
        })
        .collect())
}

/// Wheels listed on a PEP 503 "simple" index page: `<a href="name.whl#sha256=...">`.
fn parse_simple_index(html: &str, base_url: &str) -> Vec<Candidate> {
    html.split("href=\"")
        .skip(1)
        .filter_map(|rest| {
            let href = &rest[..rest.find('"')?];
            let (path, fragment) = href.split_once('#').unwrap_or((href, ""));
            let filename = path.rsplit('/').next()?.to_owned();
            let url = if path.starts_with("https://") { path.to_owned() } else { format!("{base_url}{path}") };
            Some(Candidate {
                version: wheel_version(&filename)?.to_owned(),
                sha256: fragment.strip_prefix("sha256=").and_then(fetch::normalize_sha256),
                filename,
                url,
                size: None,
            })
        })
        .collect()
}

/// Version field of a wheel file name (`{name}-{version}-{python}-{abi}-{platform}.whl`).
fn wheel_version(filename: &str) -> Option<&str> {
    filename.strip_suffix(".whl")?.split('-').nth(1)
}

/// Whether a wheel's platform tag runs on `target`.
fn wheel_supports(filename: &str, target: Target) -> bool {
    let Some(platform) = filename.strip_suffix(".whl").and_then(|f| f.rsplit('-').next()) else { return false };
    platform.split('.').any(|tag| match (target.os, target.arch) {
        (Os::Linux, Arch::X86_64) => tag.starts_with("manylinux") && tag.ends_with("_x86_64"),
        (Os::Linux, Arch::Aarch64) => tag.starts_with("manylinux") && tag.ends_with("_aarch64"),
        (Os::Windows, Arch::X86_64) => tag == "win_amd64",
        (Os::Windows, Arch::Aarch64) => tag == "win_arm64",
        (Os::Mac, Arch::Aarch64) => tag.starts_with("macosx") && tag.ends_with("_arm64"),
        (Os::Mac, Arch::X86_64) => tag.starts_with("macosx") && (tag.ends_with("_x86_64") || tag.ends_with("_intel")),
    })
}

/// Sort key of a final release (`13.4.92`); `None` for pre-releases like `13.7.0.10rc1`.
fn release_key(version: &str) -> Option<Vec<u64>> {
    version.split('.').map(|part| part.parse().ok()).collect()
}

/// Newest final release matching the version prefix with a wheel for `target`.
fn pick(candidates: Vec<Candidate>, versions: &str, target: Target) -> Option<Candidate> {
    candidates
        .into_iter()
        .filter(|c| c.version.starts_with(versions) && wheel_supports(&c.filename, target))
        .filter_map(|c| Some((release_key(&c.version)?, c)))
        .max_by(|(a, _), (b, _)| a.cmp(b))
        .map(|(_, c)| c)
}

/// PyPI package whose wheels carry one static ffmpeg build per platform (21–31 MB).
pub const FFMPEG_PACKAGE: &str = "imageio-ffmpeg";
/// Only these releases are known to keep their binary under [`FFMPEG_MEMBER_PREFIX`].
const FFMPEG_VERSIONS: &str = "0.";
/// Archive path prefix of the ffmpeg executable inside those wheels
/// (`imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`).
pub const FFMPEG_MEMBER_PREFIX: &str = "imageio_ffmpeg/binaries/ffmpeg-";

/// The newest `imageio-ffmpeg` wheel for `target`.
pub fn ffmpeg_download(target: Target) -> Result<(String, Download)> {
    let best = pick(pypi_candidates(FFMPEG_PACKAGE)?, FFMPEG_VERSIONS, target)
        .with_context(|| format!("no ffmpeg build for {:?} {:?}", target.os, target.arch))?;
    Ok((best.version, Download { url: best.url, file_name: best.filename, size: best.size, sha256: best.sha256 }))
}

/// A model with its size and SHA-256 (Hugging Face publishes both for LFS files).
pub fn model_download(asset: &Asset) -> Result<Download> {
    let (size, sha256) = fetch::head(asset.url)?;
    Ok(Download { url: asset.url.to_owned(), file_name: asset.file.to_owned(), size, sha256 })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINUX_X64: Target = Target { os: Os::Linux, arch: Arch::X86_64 };
    const WINDOWS_X64: Target = Target { os: Os::Windows, arch: Arch::X86_64 };

    fn candidate(version: &str, platform: &str) -> Candidate {
        let filename = format!("nvidia_cublas-{version}-py3-none-{platform}.whl");
        Candidate { version: version.to_owned(), url: format!("https://files/{filename}"), filename, size: Some(1), sha256: None }
    }

    #[test]
    fn ort_archive_names() {
        let name = |target, flavour| ort_archive_name(target, flavour, ORT_VERSION).unwrap();
        assert_eq!(name(LINUX_X64, Flavour::Cuda13), "onnxruntime-linux-x64-gpu_cuda13-1.28.2.tgz");
        assert_eq!(name(LINUX_X64, Flavour::Cpu), "onnxruntime-linux-x64-1.28.2.tgz");
        assert_eq!(name(WINDOWS_X64, Flavour::Cuda12), "onnxruntime-win-x64-gpu_cuda12-1.28.2.zip");
        assert_eq!(name(Target { os: Os::Mac, arch: Arch::Aarch64 }, Flavour::Coreml), "onnxruntime-osx-arm64-1.28.2.tgz");
        assert_eq!(name(Target { os: Os::Linux, arch: Arch::Aarch64 }, Flavour::Cpu), "onnxruntime-linux-aarch64-1.28.2.tgz");
        assert!(ort_archive_name(Target { os: Os::Mac, arch: Arch::X86_64 }, Flavour::Cpu, ORT_VERSION).is_err());
        assert!(ort_archive_name(Target { os: Os::Linux, arch: Arch::Aarch64 }, Flavour::Cuda13, ORT_VERSION).is_err());
    }

    #[test]
    fn wheel_platform_tags() {
        let linux = "nvidia_cufft-12.4.0.43-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
        assert!(wheel_supports(linux, LINUX_X64));
        assert!(!wheel_supports(linux, WINDOWS_X64));
        assert!(!wheel_supports(linux, Target { os: Os::Linux, arch: Arch::Aarch64 }));
        assert!(wheel_supports("nvidia_cufft-12.4.0.43-py3-none-win_amd64.whl", WINDOWS_X64));
        assert!(!wheel_supports("tensorrt_cu13_libs-11.3.0.99.tar.gz", LINUX_X64));
        let mac_arm = Target { os: Os::Mac, arch: Arch::Aarch64 };
        let mac_intel = Target { os: Os::Mac, arch: Arch::X86_64 };
        assert!(wheel_supports("imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl", mac_arm));
        assert!(!wheel_supports("imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl", mac_intel));
        assert!(wheel_supports("imageio_ffmpeg-0.6.0-py3-none-macosx_10_9_intel.macosx_10_9_x86_64.whl", mac_intel));
        assert!(!wheel_supports("imageio_ffmpeg-0.6.0-py3-none-win32.whl", WINDOWS_X64));
    }

    #[test]
    fn picks_newest_compatible_final_release() {
        let candidates = vec![
            candidate("13.8.1.7", "manylinux_2_27_x86_64"),
            candidate("13.8.0.4", "win_amd64"),
            candidate("13.10.0.1rc1", "win_amd64"),
            candidate("13.7.0.27", "win_amd64"),
            candidate("14.0.0.1", "win_amd64"),
        ];
        assert_eq!(pick(candidates.clone(), "13.", WINDOWS_X64).unwrap().version, "13.8.0.4");
        assert_eq!(pick(candidates.clone(), "13.", LINUX_X64).unwrap().version, "13.8.1.7");
        assert!(pick(candidates, "12.", LINUX_X64).is_none());
    }

    #[test]
    fn numeric_version_order() {
        assert!(release_key("13.10.0").unwrap() > release_key("13.9.5").unwrap());
        assert_eq!(release_key("10.13.3.9.post1"), None);
    }

    #[test]
    fn parses_nvidia_simple_index() {
        let hash = "044468b0705728c0c8f0e95325ba881ff3e23bd2eebce0b1e628953ff90bbb5f";
        let html = format!(
            r#"<a href="tensorrt_cu13_libs-10.13.2.6-py2.py3-none-manylinux_2_28_x86_64.whl#sha256={hash}">x</a><br>
               <a href="tensorrt_cu13_libs-10.13.3.9-py2.py3-none-win_amd64.whl">y</a>"#
        );
        let parsed = parse_simple_index(&html, "https://pypi.nvidia.com/tensorrt-cu13-libs/");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].version, "10.13.2.6");
        assert_eq!(parsed[0].sha256.as_deref(), Some(hash));
        assert_eq!(
            parsed[0].url,
            "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.13.2.6-py2.py3-none-manylinux_2_28_x86_64.whl"
        );
        assert_eq!(parsed[1].sha256, None);
    }

    #[test]
    fn wheel_sets() {
        assert_eq!(cuda_wheels(13, false).len(), CUDA13.len());
        let with_trt = cuda_wheels(12, true);
        assert_eq!(with_trt.last().unwrap().package, "tensorrt-cu12-libs");
        assert!(with_trt.iter().all(|w| w.package.contains("cu12")));
    }
}
