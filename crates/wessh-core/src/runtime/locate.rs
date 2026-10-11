//! Finding the ONNX Runtime library and the optional CUDA / TensorRT redistributables.
//!
//! Portable layout (everything relative to the ONNX Runtime library directory):
//! ```text
//! lib/libonnxruntime.so
//! lib/libonnxruntime_providers_{cuda,tensorrt,shared}.so
//! lib/cuda/     CUDA runtime, cuBLAS, cuFFT, cuRAND, cuDNN, (TensorRT) libraries
//! ```

use crate::assets::data_dir;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// File name of the ONNX Runtime shared library on this platform.
pub const RUNTIME_LIB: &str = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

/// Where `wessh setup` installs ONNX Runtime: `<data dir>/ort/lib`.
#[must_use]
pub fn default_runtime_dir() -> PathBuf {
    data_dir().join("ort").join("lib")
}

/// Resolve the ONNX Runtime library: explicit path, `ORT_DYLIB_PATH`, next to the executable,
/// then `<data dir>/ort/lib`.
pub fn runtime_library(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit.map(Path::to_path_buf).or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from)) {
        return Ok(path);
    }
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let candidates = exe_dir.into_iter().flat_map(|d| [d.join("ort").join("lib"), d.join("lib"), d]).chain([default_runtime_dir()]);
    candidates
        .map(|dir| dir.join(RUNTIME_LIB))
        .find(|p| p.exists())
        .ok_or_else(|| Error::RuntimeNotFound { searched: default_runtime_dir() })
}

/// Whether an execution provider plugin ships next to the runtime library.
#[must_use]
pub fn has_provider(runtime_dir: &Path, name: &str) -> bool {
    let file = if cfg!(target_os = "windows") {
        format!("onnxruntime_providers_{name}.dll")
    } else {
        format!("libonnxruntime_providers_{name}.so")
    };
    runtime_dir.join(file).exists()
}

/// Directory with bundled CUDA libraries (`$WESSH_CUDA_DIR`, or `cuda/` beside the runtime).
#[must_use]
pub fn cuda_dir(runtime_dir: &Path) -> Option<PathBuf> {
    std::env::var_os("WESSH_CUDA_DIR").map(PathBuf::from).or_else(|| Some(runtime_dir.join("cuda"))).filter(|d| d.is_dir())
}

/// Whether TensorRT's core library is loadable from the bundled CUDA dir.
#[must_use]
pub fn has_tensorrt_runtime(cuda_dir: Option<&Path>) -> bool {
    cuda_dir.is_some_and(|d| {
        std::fs::read_dir(d).is_ok_and(|entries| {
            entries.flatten().any(|e| {
                e.file_name().to_string_lossy().starts_with("libnvinfer.so") || e.file_name().to_string_lossy().starts_with("nvinfer")
            })
        })
    })
}

/// Preload every shared library in `dir` so the CUDA provider finds them without
/// `LD_LIBRARY_PATH`. Libraries are retried until their own dependencies have been loaded.
#[allow(clippy::must_use_candidate, reason = "the count is diagnostic; loading is the point")]
pub fn preload_libraries(dir: &Path) -> usize {
    let is_library = |p: &Path| {
        let name = p.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        let extension = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase());
        name.contains(".so") || matches!(extension.as_deref(), Some("dll" | "dylib"))
    };
    let mut pending: Vec<PathBuf> =
        std::fs::read_dir(dir).map(|entries| entries.flatten().map(|e| e.path()).filter(|p| is_library(p)).collect()).unwrap_or_default();
    pending.sort();
    let mut loaded = 0;
    loop {
        let before = pending.len();
        pending.retain(|lib| ort::util::preload_dylib(lib).is_err());
        loaded += before - pending.len();
        if pending.is_empty() || pending.len() == before {
            return loaded;
        }
    }
}
