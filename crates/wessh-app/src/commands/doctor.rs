//! `wessh doctor`: report what is installed and how to fix what isn't.

use crate::cli::Options;
use crate::install::archive::is_shared_library;
use crate::install::platform::{self, Flavour, Os, Target};
use crate::install::sources;
use anyhow::Result;
use ort::ep::{CUDA, ExecutionProvider, TensorRT};
use std::fmt::Display;
use std::path::Path;
use std::time::Instant;
use wessh_core::assets::{self, data_dir};
use wessh_core::face::Detector;
use wessh_core::runtime::locate;
use wessh_core::{Device, Engine, Frame};
use wessh_media::WindowsCameraStatus;

const V4L2LOOPBACK_LOAD: &str = "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=wessh exclusive_caps=1";

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
    row("wessh", env!("CARGO_PKG_VERSION"));
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

    check_runtime(options, target, recommended.flavour, &mut report);
    check_ffmpeg();
    check_models(options, &mut report);
    if target.os == Os::Linux {
        check_v4l2loopback(&mut report);
    }
    if target.os == Os::Windows {
        check_windows_camera(&mut report);
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

fn check_runtime(options: &Options, target: Target, recommended: Flavour, report: &mut Report) {
    let wanted_cuda = recommended.cuda_major();
    let library = match locate::runtime_library(options.ort.as_deref()) {
        Ok(library) => library,
        Err(e) => {
            row("ONNX Runtime", "not found");
            report.problems.push(format!("{e}. Run `wessh setup`"));
            return;
        }
    };
    let runtime_dir = library.parent().unwrap_or(Path::new("."));
    let mut plugins: Vec<&str> = locate::BUNDLED_PROVIDERS.into_iter().filter(|p| locate::has_provider(runtime_dir, p)).collect();
    plugins.extend(
        locate::Plugin::ALL.into_iter().filter(|&p| locate::plugin_library(runtime_dir, p).is_some()).map(locate::Plugin::dir_name),
    );
    if let Some(plugin) = recommended.plugin()
        && locate::plugin_library(runtime_dir, plugin).is_none()
        && plugins.is_empty()
    {
        report.problems.push(format!("the {recommended} plugin is not installed, so wessh runs on the CPU; run `wessh setup`"));
    }

    let cuda_dir = locate::cuda_dir(runtime_dir);
    if let Some(dir) = &cuda_dir {
        let total = std::fs::read_dir(dir)
            .map_or(0, |entries| entries.flatten().filter(|e| is_shared_library(&e.file_name().to_string_lossy())).count());
        let loaded = locate::preload_libraries(dir);
        row("CUDA libraries", format!("{} ({loaded} of {total} load)", dir.display()));
        if loaded < total {
            report.problems.push("some bundled CUDA libraries fail to load (missing dependencies); run `wessh setup --force`".to_owned());
        }
    }
    if let Some(major) = wanted_cuda {
        let missing: Vec<&str> = sources::cuda_wheels_for_runtime(major, sources::ORT_VERSION)
            .into_iter()
            .filter(|w| !cuda_dir.as_deref().is_some_and(|d| w.is_installed(d, target.os)))
            .map(|w| w.package)
            .collect();
        if !plugins.contains(&"cuda") {
            report
                .problems
                .push("the installed ONNX Runtime is a CPU-only build although an NVIDIA GPU is usable; run `wessh setup`".to_owned());
        } else if !missing.is_empty() {
            row("CUDA missing", missing.join(", "));
            report.problems.push("CUDA libraries are missing, so wessh would fall back to the CPU; run `wessh setup`".to_owned());
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
            row("plugins", if plugins.is_empty() { "none (CPU only)".to_owned() } else { plugins.join(", ") });
            row("backend", format!("{} (with --device auto)", engine.backend()));
            if let Some(reason) = engine.gpu_failure() {
                report
                    .problems
                    .push(format!("inference falls back to the CPU: {reason}. Run `wessh setup` to install the runtime for this machine"));
            }
            check_inference(&engine, report);
        }
        Err(e) => {
            row("ONNX Runtime", library.display());
            report.problems.push(format!("{e}. Run `wessh setup --force` to reinstall ONNX Runtime {}", sources::ORT_VERSION));
        }
    }
}

/// Detection runs this many times after the first, for the time per frame.
const INFERENCE_RUNS: u32 = 20;

/// Run a model for real: a runtime can load and still fail (or crawl) once it has to compute.
fn check_inference(engine: &Engine, report: &mut Report) {
    if !assets::DETECTOR.path().exists() {
        row("inference", "not checked (the face detector isn't downloaded yet)");
        return;
    }
    let started = Instant::now();
    let measured = Detector::new(engine).and_then(|mut detector| {
        let loaded = started.elapsed();
        let frame = Frame::new(1280, 720);
        detector.detect(&frame)?;
        let started = Instant::now();
        for _ in 0..INFERENCE_RUNS {
            detector.detect(&frame)?;
        }
        Ok((loaded, started.elapsed() / INFERENCE_RUNS))
    });
    match measured {
        Ok((loaded, per_frame)) => row(
            "inference",
            format!(
                "face detection {:.1} ms a frame on {} (loaded in {:.1} s)",
                per_frame.as_secs_f64() * 1e3,
                engine.backend(),
                loaded.as_secs_f64()
            ),
        ),
        Err(e) => {
            row("inference", "fails");
            report.problems.push(format!("the face detector fails on {}: {e}", engine.backend()));
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

/// Only video files and network cameras need ffmpeg, and the app downloads it when they do.
fn check_ffmpeg() {
    row("ffmpeg", platform::ffmpeg_version().unwrap_or_else(|| "not installed (downloaded when a video is first opened)".to_owned()));
}

fn check_models(options: &Options, report: &mut Report) {
    let (present, absent): (Vec<_>, Vec<_>) = assets::required(&options.settings()).into_iter().partition(assets::Asset::is_downloaded);
    let names = |list: &[assets::Asset]| list.iter().map(|a| a.name).collect::<Vec<_>>().join(", ");
    row("models", if present.is_empty() { "none downloaded".to_owned() } else { names(&present) });
    if !absent.is_empty() {
        row("not downloaded", names(&absent));
    }
    if !absent.is_empty() {
        report.problems.push("some models are missing: run `wessh setup`, or start the app to download them".to_owned());
    }
}

fn check_windows_camera(report: &mut Report) {
    let status = wessh_media::windows_camera::status();
    row("virtual camera", super::camera::describe(status));
    match status {
        WindowsCameraStatus::NotInstalled => {
            report
                .suggestions
                .push("to appear as a webcam in other apps, run `wessh camera install` (asks for administrator rights once)".to_owned());
        }
        WindowsCameraStatus::Outdated => {
            report.problems.push("the virtual camera is from another wessh version: run `wessh camera install`".to_owned());
        }
        WindowsCameraStatus::Unsupported | WindowsCameraStatus::Ready => {}
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
