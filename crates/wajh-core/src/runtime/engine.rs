use super::{Model, locate};
use crate::assets::{Asset, data_dir};
use crate::{Error, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch, TensorRT, cuda::ConvAlgorithmSearch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;

/// Requested inference backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum Device {
    /// Fastest available: TensorRT, then CUDA, then CPU.
    #[default]
    Auto,
    Tensorrt,
    Cuda,
    Cpu,
}

/// Backend actually in use after probing what's installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    TensorRt,
    Cuda,
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            Self::Cpu => "CPU",
        })
    }
}

/// Initialised ONNX Runtime plus the execution-provider policy used for every model.
#[derive(Clone, Debug)]
pub struct Engine {
    backend: Backend,
    cpu_threads: usize,
}

impl Engine {
    /// Load ONNX Runtime (and bundled CUDA libraries) and pick the best backend for `device`.
    pub fn init(runtime: Option<&Path>, device: Device) -> Result<Self> {
        let library = locate::runtime_library(runtime)?;
        let runtime_dir = library.parent().unwrap_or(Path::new("."));
        let cuda_dir = locate::cuda_dir(runtime_dir);
        let wants_gpu = device != Device::Cpu;
        if wants_gpu && let Some(dir) = &cuda_dir {
            locate::preload_libraries(dir);
        }
        ort::init_from(&library)
            .map_err(|e| Error::RuntimeLoad { path: library.clone(), reason: e.to_string() })?
            .with_name("wajh")
            .commit();

        let cuda = wants_gpu && locate::has_provider(runtime_dir, "cuda");
        let tensorrt = cuda
            && matches!(device, Device::Auto | Device::Tensorrt)
            && locate::has_provider(runtime_dir, "tensorrt")
            && locate::has_tensorrt_runtime(cuda_dir.as_deref());
        let backend = match (tensorrt, cuda) {
            (true, _) => Backend::TensorRt,
            (false, true) => Backend::Cuda,
            _ => Backend::Cpu,
        };
        Ok(Self { backend, cpu_threads: cpu_threads() })
    }

    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Load `asset` (downloading it if needed) with this engine's execution providers.
    pub fn load(&self, asset: &Asset) -> Result<Model> {
        let path = asset.ensure()?;
        let session = self.session(&path).map_err(|source| Error::ModelLoad { path: path.clone(), source })?;
        Ok(Model::new(session, asset.name))
    }

    fn session(&self, path: &Path) -> ort::Result<Session> {
        let gpu = self.backend != Backend::Cpu;
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(ort::Error::<()>::from)?
            .with_intra_threads(if gpu { 1 } else { self.cpu_threads })
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers(self.execution_providers())
            .map_err(ort::Error::<()>::from)?
            .commit_from_file(path)
    }

    fn execution_providers(&self) -> Vec<ExecutionProviderDispatch> {
        let cache = data_dir().join("trt_cache");
        let tensorrt = || {
            let _ = std::fs::create_dir_all(&cache);
            let cache = cache.to_string_lossy().into_owned();
            TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(&cache)
                .with_timing_cache(true)
                .with_timing_cache_path(&cache)
                .build()
        };
        let cuda = || {
            CUDA::default()
                .with_conv_algorithm_search(ConvAlgorithmSearch::Exhaustive)
                .with_conv_max_workspace(true)
                .with_tf32(true)
                .build()
        };
        match self.backend {
            Backend::TensorRt => vec![tensorrt(), cuda()],
            Backend::Cuda => vec![cuda()],
            Backend::Cpu => Vec::new(),
        }
    }
}

/// ONNX Runtime scales best with one thread per physical core; `WAJH_THREADS` overrides.
fn cpu_threads() -> usize {
    std::env::var("WAJH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (std::thread::available_parallelism().map_or(8, std::num::NonZero::get) / 2).max(1))
}
