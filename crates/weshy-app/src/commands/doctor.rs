//! `weshy doctor`: report what is installed and how to fix what isn't.

use crate::cli::Options;
use crate::install::archive::is_shared_library;
use crate::install::platform::{self, Os, Target};
use crate::install::sources;
use anyhow::Result;
use ort::ep::{CUDA, ExecutionProvider, TensorRT};
use std::fmt::Display;
use std::path::Path;
use weshy_core::assets::{self, data_dir};
use weshy_core::runtime::locate;
use weshy_core::{Device, Engine};

const V4L2LOOPBACK_LOAD: &str = "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=weshy exclusive_caps=1";

/// Findings that need action.
#[derive(Default)]
struct Report {
    problems: Vec<String>,
    suggestions: Vec<String>,
}

fn row(label: &str, value: impl Display) {
    println!("{label:<16}{value}");
}

pub fn run(options: &Options) -> Result<()> {
    let target = Target::host()?;
    let mut report = Report::default();
    row("weshy", env!("CARGO_PKG_VERSION"));
    row("data dir", data_dir().display());

    let gpus = platform::detect_gpus(target);
    if gpus.is_empty() {
        row("GPU", "none detected");
    }
    for gpu in &gpus {
        row("GPU", gpu);
    }
    let recommended = platform::choose_flavour(target, &gpus);
    row("best runtime", format!("{} ({})", recommended.flavour, recommended.reason));

    check_runtime(options, target, recommended.flavour.cuda_major(), &mut report);
    check_ffmpeg(target, &mut report);
    check_models(options, &mut report);
    if target.os == Os::Linux {
        check_v4l2loopback(&mut report);
    }

    println!();
    if report.problems.is_empty() {
        println!("no problems found");
    } else {
        println!("to fix:");
        report.problems.iter().for_each(|p| println!("  - {p}"));
    }
    if !report.suggestions.is_empty() {
        println!("optional:");
        report.suggestions.iter().for_each(|s| println!("  - {s}"));
    }
    Ok(())
}

fn check_runtime(options: &Options, target: Target, wanted_cuda: Option<u32>, report: &mut Report) {
    let library = match locate::runtime_library(options.ort.as_deref()) {
        Ok(library) => library,
        Err(e) => {
            row("ONNX Runtime", "not found");
            report.problems.push(format!("{e}. Run `weshy setup`"));
            return;
        }
    };
    let runtime_dir = library.parent().unwrap_or(Path::new("."));
    let plugins: Vec<&str> = ["cuda", "tensorrt"].into_iter().filter(|p| locate::has_provider(runtime_dir, p)).collect();

    let cuda_dir = locate::cuda_dir(runtime_dir);
    if let Some(dir) = &cuda_dir {
        let total = std::fs::read_dir(dir)
            .map_or(0, |entries| entries.flatten().filter(|e| is_shared_library(&e.file_name().to_string_lossy())).count());
        let loaded = locate::preload_libraries(dir);
        row("CUDA libraries", format!("{} ({loaded} of {total} load)", dir.display()));
        if loaded < total {
            report.problems.push("some bundled CUDA libraries fail to load (missing dependencies); run `weshy setup --force`".to_owned());
        }
    }
    if let Some(major) = wanted_cuda {
        let missing: Vec<&str> = sources::cuda_wheels(major, false)
            .into_iter()
            .filter(|w| !cuda_dir.as_deref().is_some_and(|d| w.is_installed(d, target.os)))
            .map(|w| w.package)
            .collect();
        if plugins.is_empty() {
            report
                .problems
                .push("the installed ONNX Runtime is a CPU-only build although an NVIDIA GPU is usable; run `weshy setup`".to_owned());
        } else if !missing.is_empty() {
            row("CUDA missing", missing.join(", "));
            report.problems.push("CUDA libraries are missing, so weshy would fall back to the CPU; run `weshy setup`".to_owned());
        }
    }

    match Engine::init(Some(&library), Device::Auto) {
        Ok(engine) => {
            row("ONNX Runtime", format!("{} ({})", runtime_version(&library, ort::info()), library.display()));
            let providers: Vec<&str> = [("CUDA", CUDA::default().is_available()), ("TensorRT", TensorRT::default().is_available())]
                .into_iter()
                .filter_map(|(name, available)| available.unwrap_or(false).then_some(name))
                .chain(["CPU"])
                .collect();
            row("providers", providers.join(", "));
            row("plugins", if plugins.is_empty() { "none (CPU-only build)".to_owned() } else { plugins.join(", ") });
            row("backend", format!("{} (with --device auto)", engine.backend()));
            let has_tensorrt = plugins.contains(&"tensorrt") && locate::has_tensorrt_runtime(cuda_dir.as_deref());
            if wanted_cuda.is_some() && !plugins.is_empty() && !has_tensorrt {
                report.suggestions.push("`weshy setup --tensorrt` adds TensorRT, the fastest NVIDIA backend".to_owned());
            }
        }
        Err(e) => {
            row("ONNX Runtime", library.display());
            report.problems.push(format!("{e}. Run `weshy setup --force` to reinstall ONNX Runtime {}", sources::ORT_VERSION));
        }
    }
}

/// ONNX Runtime's version: from the versioned file a `libonnxruntime.so` symlink points to
/// (`libonnxruntime.so.1.28.2`, `libonnxruntime.1.28.2.dylib`), else from its build info
/// (`git-branch=rel-1.28.2`); release builds report `git-branch=HEAD`, so the commit is the last resort.
fn runtime_version(library: &Path, build_info: &str) -> String {
    let resolved = std::fs::canonicalize(library).unwrap_or_else(|_| library.to_path_buf());
    let file_name = resolved.file_name().unwrap_or_default().to_string_lossy();
    let field = |key: &str| build_info.split([',', ' ']).find_map(|f| f.strip_prefix(key)).map(str::to_owned);
    version_from_file_name(&file_name)
        .map(str::to_owned)
        .or_else(|| field("git-branch=rel-"))
        .or_else(|| field("git-commit-id=").map(|commit| format!("commit {commit}")))
        .unwrap_or_else(|| "unknown version".to_owned())
}

fn version_from_file_name(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("libonnxruntime.")?;
    let version = rest.strip_prefix("so.").or_else(|| rest.strip_suffix(".dylib"))?;
    (version.contains('.') && version.bytes().all(|b| b.is_ascii_digit() || b == b'.')).then_some(version)
}

fn check_ffmpeg(target: Target, report: &mut Report) {
    if let Some(version) = platform::ffmpeg_version() {
        row("ffmpeg", version);
    } else {
        row("ffmpeg", "not found");
        report.problems.push(format!("ffmpeg is needed for camera and video modes: {}", platform::ffmpeg_install_hint(target.os)));
    }
}

fn check_models(options: &Options, report: &mut Report) {
    let (present, absent): (Vec<_>, Vec<_>) = assets::required(&options.settings()).into_iter().partition(assets::Asset::is_downloaded);
    let names = |list: &[assets::Asset]| list.iter().map(|a| a.name).collect::<Vec<_>>().join(", ");
    row("models", if present.is_empty() { "none downloaded".to_owned() } else { names(&present) });
    if !absent.is_empty() {
        row("not downloaded", names(&absent));
    }
    if !absent.is_empty() {
        report.problems.push("some models are missing: run `weshy setup`, or start the app to download them".to_owned());
    }
}

fn check_v4l2loopback(report: &mut Report) {
    if !Path::new("/sys/module/v4l2loopback").exists() {
        row("virtual camera", "v4l2loopback not loaded");
        let installed = platform::run_stdout("modinfo", &["v4l2loopback"]).is_some();
        report.suggestions.push(if installed {
            format!("for virtual camera output, load v4l2loopback: {V4L2LOOPBACK_LOAD}")
        } else {
            format!("for virtual camera output, install v4l2loopback (package v4l2loopback-dkms or similar), then: {V4L2LOOPBACK_LOAD}")
        });
        return;
    }
    let devices: Vec<String> = std::fs::read_dir("/sys/devices/virtual/video4linux")
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    let name = std::fs::read_to_string(e.path().join("name")).unwrap_or_default();
                    format!("/dev/{} ({})", e.file_name().to_string_lossy(), name.trim())
                })
                .collect()
        })
        .unwrap_or_default();
    if devices.is_empty() {
        row("virtual camera", "v4l2loopback loaded but has no devices");
        report.suggestions.push(format!("reload v4l2loopback with a device: sudo modprobe -r v4l2loopback && {V4L2LOOPBACK_LOAD}"));
    } else {
        row("virtual camera", devices.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_version_from_file_name() {
        assert_eq!(version_from_file_name("libonnxruntime.so.1.28.2"), Some("1.28.2"));
        assert_eq!(version_from_file_name("libonnxruntime.1.28.2.dylib"), Some("1.28.2"));
        assert_eq!(version_from_file_name("libonnxruntime.so"), None);
        assert_eq!(version_from_file_name("onnxruntime.dll"), None);
    }

    #[test]
    fn runtime_version_from_build_info() {
        let dll = Path::new("onnxruntime.dll");
        assert_eq!(runtime_version(dll, "ORT Build Info: git-branch=rel-1.28.2, git-commit-id=26250ae, build type=Release"), "1.28.2");
        assert_eq!(
            runtime_version(dll, "ORT Build Info: git-branch=HEAD, git-commit-id=33ca962823, build type=Release"),
            "commit 33ca962823"
        );
    }
}
