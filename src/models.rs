//! Model registry, first-run downloads and locating the ONNX Runtime shared library.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct ModelFile {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
}

macro_rules! ff {
    ($name:literal, $repo:literal, $file:literal) => {
        ModelFile { name: $name, file: $file, url: concat!("https://huggingface.co/facefusion/", $repo, "/resolve/main/", $file) }
    };
}

pub const DETECTOR: ModelFile = ff!("scrfd_2.5g", "models-3.0.0", "scrfd_2.5g.onnx");
pub const RECOGNIZER: ModelFile = ff!("arcface_w600k_r50", "models-3.0.0", "arcface_w600k_r50.onnx");
pub const ENHANCER: ModelFile = ff!("gpen_bfr_256", "models-3.0.0", "gpen_bfr_256.onnx");

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SwapModel {
    /// InsightFace inswapper 128px (fp16). Fastest.
    Inswapper,
    /// FaceFusion HyperSwap 1a, 256px.
    Hyperswap1a,
    /// FaceFusion HyperSwap 1b, 256px.
    Hyperswap1b,
    /// FaceFusion HyperSwap 1c, 256px. Best identity transfer.
    Hyperswap1c,
}

impl SwapModel {
    pub const ALL: [SwapModel; 4] = [SwapModel::Inswapper, SwapModel::Hyperswap1a, SwapModel::Hyperswap1b, SwapModel::Hyperswap1c];

    pub fn file(self) -> ModelFile {
        match self {
            SwapModel::Inswapper => ff!("inswapper_128_fp16", "models-3.0.0", "inswapper_128_fp16.onnx"),
            SwapModel::Hyperswap1a => ff!("hyperswap_1a_256", "models-3.3.0", "hyperswap_1a_256.onnx"),
            SwapModel::Hyperswap1b => ff!("hyperswap_1b_256", "models-3.3.0", "hyperswap_1b_256.onnx"),
            SwapModel::Hyperswap1c => ff!("hyperswap_1c_256", "models-3.3.0", "hyperswap_1c_256.onnx"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SwapModel::Inswapper => "inswapper 128 (fast)",
            SwapModel::Hyperswap1a => "HyperSwap 1a 256",
            SwapModel::Hyperswap1b => "HyperSwap 1b 256",
            SwapModel::Hyperswap1c => "HyperSwap 1c 256",
        }
    }
}

pub fn data_dir() -> PathBuf {
    std::env::var_os("WAJH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".")).join("wajh"))
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Path to a model, downloading it (resumable, via curl) if it's missing.
pub fn ensure(model: &ModelFile) -> Result<PathBuf> {
    let dir = models_dir();
    let path = dir.join(model.file);
    if path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir)?;
    let part = dir.join(format!("{}.part", model.file));
    eprintln!("downloading {} ...", model.name);
    let status = Command::new("curl")
        .args(["-fL", "--retry", "5", "-C", "-", "--progress-bar", "-o"])
        .arg(&part)
        .arg(model.url)
        .status()
        .context("running curl")?;
    if !status.success() {
        bail!("download of {} failed ({status})", model.url);
    }
    std::fs::rename(&part, &path)?;
    Ok(path)
}

fn lib_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so"]
    }
}

fn find_lib_in(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut hits: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            lib_names().iter().any(|l| n.starts_with(l))
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Locate the ONNX Runtime library: `--ort` flag, `ORT_DYLIB_PATH`, next to the executable,
/// or `<data dir>/ort/lib`.
pub fn find_onnxruntime(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
        return Ok(PathBuf::from(p));
    }
    let mut dirs = Vec::new();
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir.join("ort"));
        dirs.push(exe_dir);
    }
    dirs.push(data_dir().join("ort").join("lib"));
    dirs.push(data_dir().join("ort"));
    for d in &dirs {
        if let Some(p) = find_lib_in(d) {
            return Ok(p);
        }
    }
    bail!(
        "ONNX Runtime library not found. Put libonnxruntime next to the binary, in {}, or set ORT_DYLIB_PATH",
        data_dir().join("ort/lib").display()
    )
}
