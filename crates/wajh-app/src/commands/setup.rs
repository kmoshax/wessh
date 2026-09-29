//! `wajh setup`: one command that makes a machine ready to run wajh.
//!
//! Picks an ONNX Runtime flavour for the GPU and driver, installs the runtime into
//! `<data dir>/ort/lib`, NVIDIA's CUDA / cuDNN (and optionally TensorRT) libraries from their
//! Python wheels into `<data dir>/ort/lib/cuda`, and the default models into
//! `<data dir>/models`. Every step is skipped when already done, downloads resume, and
//! checksums are verified whenever the source publishes one.

pub mod archive;
mod fetch;
pub mod platform;
pub mod sources;

use crate::cli::{Options, SetupArgs};
use anyhow::{Result, bail};
use fetch::Download;
use platform::{Choice, Flavour, Target};
use std::path::PathBuf;
use wajh_core::assets::data_dir;
use wajh_core::runtime::locate;

/// Where a component ends up.
enum Install {
    /// Shared libraries extracted from an archive into this directory.
    Libraries(PathBuf),
    /// A single file downloaded to this path.
    File(PathBuf),
}

struct Step {
    name: String,
    version: String,
    install: Install,
    /// `None` when already installed.
    download: Option<Download>,
}

pub fn run(args: &SetupArgs, options: &Options) -> Result<()> {
    let target = Target::host()?;
    let gpus = platform::detect_gpus(target);
    let choice = match args.flavour {
        Some(flavour) => Choice { flavour, reason: "chosen with --flavour".to_owned() },
        None => platform::choose_flavour(target, &gpus),
    };
    if args.tensorrt && choice.flavour.cuda_major().is_none() {
        bail!("--tensorrt needs an NVIDIA GPU and a CUDA flavour (selected: {})", choice.flavour);
    }

    println!("data dir   {}", data_dir().display());
    if gpus.is_empty() {
        println!("GPU        none detected");
    }
    for gpu in &gpus {
        println!("GPU        {gpu}");
    }
    println!("runtime    {} ({})", choice.flavour, choice.reason);
    println!("\nchecking what is needed...");

    let steps = plan(args, options, target, choice.flavour)?;
    print_plan(&steps);
    if !args.dry_run {
        install(&steps)?;
    }

    println!();
    match platform::ffmpeg_version() {
        Some(version) => println!("ffmpeg     {version}"),
        None => println!(
            "ffmpeg     not found; video and camera modes need it. Install it with:\n           {}",
            platform::ffmpeg_install_hint(target.os)
        ),
    }
    if args.dry_run {
        println!("\ndry run: nothing was downloaded or changed");
    } else {
        println!("\nready. `wajh doctor` checks the installation; `wajh` opens the app");
    }
    Ok(())
}

fn plan(args: &SetupArgs, options: &Options, target: Target, flavour: Flavour) -> Result<Vec<Step>> {
    let pending = |ready: bool| args.force || !ready;
    let runtime_dir = locate::default_runtime_dir();
    let mut steps = Vec::new();

    let runtime_ready = runtime_dir.join(locate::RUNTIME_LIB).exists()
        && (flavour.cuda_major().is_none() || locate::has_provider(&runtime_dir, "cuda"))
        && (!args.tensorrt || locate::has_provider(&runtime_dir, "tensorrt"));
    steps.push(Step {
        name: format!("ONNX Runtime ({flavour})"),
        version: if pending(runtime_ready) { args.ort_version.clone() } else { String::new() },
        install: Install::Libraries(runtime_dir.clone()),
        download: pending(runtime_ready).then(|| sources::ort_release(target, flavour, &args.ort_version)).transpose()?,
    });

    if let Some(cuda_major) = flavour.cuda_major() {
        let cuda_dir = runtime_dir.join("cuda");
        for wheel in sources::cuda_wheels(cuda_major, args.tensorrt) {
            let (version, download) = if pending(wheel.is_installed(&cuda_dir, target.os)) {
                let (version, download) = wheel.resolve(target)?;
                (version, Some(download))
            } else {
                (String::new(), None)
            };
            steps.push(Step { name: wheel.package.to_owned(), version, install: Install::Libraries(cuda_dir.clone()), download });
        }
    }

    if !args.no_models {
        for asset in super::models::needed(options) {
            let path = asset.path();
            let download = pending(path.exists()).then(|| sources::model_download(&asset)).transpose()?;
            steps.push(Step { name: asset.file.to_owned(), version: String::new(), install: Install::File(path), download });
        }
    }
    Ok(steps)
}

fn print_plan(steps: &[Step]) {
    let width = steps.iter().map(|s| s.name.len()).max().unwrap_or_default();
    for step in steps {
        let (size, action) = match &step.download {
            Some(download) => (download.size.map_or_else(|| "size unknown".to_owned(), human_size), "download"),
            None => (String::new(), "installed"),
        };
        println!("  {:width$}  {:12}  {size:>12}  {action}", step.name, step.version);
    }
    let downloads: Vec<&Download> = steps.iter().filter_map(|s| s.download.as_ref()).collect();
    if downloads.is_empty() {
        println!("everything is already installed (use --force to reinstall)");
    } else {
        let total: u64 = downloads.iter().filter_map(|d| d.size).sum();
        println!("{} download(s), {} in total", downloads.len(), human_size(total));
    }
}

fn install(steps: &[Step]) -> Result<()> {
    let archives = data_dir().join("downloads");
    let pending: Vec<(&Step, &Download)> = steps.iter().filter_map(|s| Some((s, s.download.as_ref()?))).collect();
    for (index, (step, download)) in pending.iter().enumerate() {
        let size = download.size.map(human_size).unwrap_or_default();
        println!("\n[{}/{}] {} {} {size}", index + 1, pending.len(), step.name, step.version);
        match &step.install {
            Install::File(path) => fetch::download(download, path)?,
            Install::Libraries(dir) => {
                let archive = archives.join(&download.file_name);
                fetch::download(download, &archive)?;
                let installed = archive::extract_libraries(&archive, dir)?;
                std::fs::remove_file(&archive)?;
                println!("      {} libraries -> {}", installed.len(), dir.display());
            }
        }
    }
    // Only leftovers of interrupted downloads could remain; an empty directory is noise.
    let _ = std::fs::remove_dir(&archives);
    Ok(())
}

/// `1536` -> `1.5 KiB`.
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    const STEP: f64 = 1024.0;
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= STEP && unit < UNITS.len() - 1 {
        value /= STEP;
        unit += 1;
    }
    if unit == 0 { format!("{bytes} B") } else { format!("{value:.1} {}", UNITS[unit]) }
}

#[cfg(test)]
mod tests {
    use super::human_size;

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1536), "1.5 KiB");
        assert_eq!(human_size(240_868_705), "229.7 MiB");
        assert_eq!(human_size(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }
}
