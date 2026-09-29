use super::binding::{Placement, StaticModel};
use super::proto::{self, GraphPatch};
use super::{Model, locate};
use crate::assets::{Asset, data_dir};
use crate::{Error, Result};
use ort::ep::coreml::{ComputeUnits, ModelFormat};
use ort::ep::{CUDA, CoreML, ExecutionProvider, ExecutionProviderDispatch, TensorRT, cuda::ConvAlgorithmSearch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;

/// Fixed sizes for a model's symbolic dimensions, and graph edits applied before loading it
/// (see [`Engine::load_static_with`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticShapes<'a> {
    /// Applied to the ONNX file before ONNX Runtime sees it: pinned input shapes, a reduced
    /// output, pruned heads.
    pub patch: Option<&'a GraphPatch<'a>>,
    /// `(dimension name, size)` overrides applied to the graph's free dimensions.
    pub dimensions: &'a [(&'a str, i64)],
    /// Shapes of the outputs, in order, for outputs whose shape metadata stays symbolic.
    pub outputs: &'a [Vec<usize>],
    /// Run on the CUDA provider even on the TensorRT backend, for models TensorRT gets wrong
    /// or doesn't speed up.
    pub skip_tensorrt: bool,
}

/// Runs made before a static model is handed out: cuDNN autotuning, then CUDA graph capture.
const WARMUP_RUNS: usize = 3;

/// How one session is set up.
#[derive(Clone, Debug)]
struct SessionOptions<'a> {
    /// Capture the whole network in a CUDA graph.
    cuda_graph: bool,
    /// Offer the network to TensorRT first (on the TensorRT backend).
    tensorrt: bool,
    /// Prefix of the model's TensorRT engine cache files. ONNX Runtime names cached engines
    /// after a hash of the graph that ignores dimension overrides, so without a prefix naming
    /// the pinned shapes, a detector loaded at 192 px would pick up the engine built at 640 px
    /// and fail with a static dimension mismatch.
    cache_prefix: &'a str,
}

/// Requested inference backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum Device {
    /// Fastest available: TensorRT, then CUDA, then CoreML (macOS), then CPU.
    #[default]
    Auto,
    Tensorrt,
    Cuda,
    /// Apple's CoreML (GPU / Neural Engine), macOS only.
    Coreml,
    Cpu,
}

/// Backend actually in use after probing what's installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    TensorRt,
    Cuda,
    CoreMl,
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
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
        let wants_cuda = matches!(device, Device::Auto | Device::Tensorrt | Device::Cuda);
        if wants_cuda && let Some(dir) = &cuda_dir {
            locate::preload_libraries(dir);
        }
        ort::init_from(&library)
            .map_err(|e| Error::RuntimeLoad { path: library.clone(), reason: e.to_string() })?
            .with_name("wajh")
            .commit();

        let cuda = wants_cuda && locate::has_provider(runtime_dir, "cuda");
        let tensorrt = cuda
            && matches!(device, Device::Auto | Device::Tensorrt)
            && locate::has_provider(runtime_dir, "tensorrt")
            && locate::has_tensorrt_runtime(cuda_dir.as_deref());
        // CoreML ships inside the macOS ONNX Runtime library rather than as a plugin.
        let coreml = cfg!(target_os = "macos")
            && matches!(device, Device::Auto | Device::Coreml)
            && CoreML::default().is_available().unwrap_or(false);
        let backend = if tensorrt {
            Backend::TensorRt
        } else if cuda {
            Backend::Cuda
        } else if coreml {
            Backend::CoreMl
        } else {
            Backend::Cpu
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
        let options = SessionOptions { cuda_graph: false, tensorrt: true, cache_prefix: &cache_prefix(asset, &[], false) };
        let session = self.session(&path, None, &options, &[]).map_err(|source| Error::ModelLoad { path: path.clone(), source })?;
        Ok(Model::new(session, asset.name))
    }

    /// Load a model whose inputs and outputs all have fixed shapes, with pre-allocated bound
    /// buffers and — on CUDA / TensorRT — a CUDA graph when the network can be captured.
    pub fn load_static(&self, asset: &Asset) -> Result<StaticModel> {
        self.load_static_with(asset, &StaticShapes::default())
    }

    /// [`Engine::load_static`] for a model with symbolic dimensions, pinned by `shapes`.
    ///
    /// The model is warmed up before it is returned, so the first frame is not slow. Capture
    /// failures (a node that falls back to the CPU, for instance) surface during the warm-up and
    /// the model is reloaded without a graph. On the TensorRT backend a model TensorRT can't
    /// build or run is reloaded on plain CUDA.
    pub fn load_static_with(&self, asset: &Asset, shapes: &StaticShapes<'_>) -> Result<StaticModel> {
        let path = asset.ensure()?;
        let patched = shapes.patch.map(|patch| proto::patch_graph(&std::fs::read(&path)?, patch)).transpose()?;
        let prefix = cache_prefix(asset, shapes.dimensions, patched.is_some());
        let load = |cuda_graph: bool, tensorrt: bool| -> Result<StaticModel> {
            let options = SessionOptions { cuda_graph, tensorrt, cache_prefix: &prefix };
            let session = self
                .session(&path, patched.as_deref(), &options, shapes.dimensions)
                .map_err(|source| Error::ModelLoad { path: path.clone(), source })?;
            let mut model = StaticModel::new(session, asset.name, self.placement(), shapes.outputs)?;
            for _ in 0..WARMUP_RUNS {
                model.run()?;
            }
            Ok(model)
        };
        let providers: &[bool] = if self.backend == Backend::TensorRt && !shapes.skip_tensorrt { &[true, false] } else { &[false] };
        let mut last = None;
        for &tensorrt in providers {
            // The TensorRT provider's CUDA graph replays without refreshing our host-bound
            // outputs (the detector then sees all-zero scores and finds no face), and TensorRT
            // already fuses the network into few kernels; graphs are only used on plain CUDA.
            if self.cuda_graphs() && !tensorrt {
                match load(true, tensorrt) {
                    Ok(model) => return Ok(model),
                    Err(e @ Error::DynamicShape { .. }) => return Err(e),
                    Err(e) => eprintln!("{}: CUDA graph unavailable ({e}); running without", asset.name),
                }
            }
            match load(false, tensorrt) {
                Ok(model) => return Ok(model),
                Err(e @ Error::DynamicShape { .. }) => return Err(e),
                Err(e) if tensorrt => eprintln!("{}: TensorRT unavailable ({e}); running on CUDA", asset.name),
                Err(e) => last = Some(e),
            }
        }
        Err(last.expect("the last provider's error is kept"))
    }

    const fn placement(&self) -> Placement {
        match self.backend {
            Backend::Cpu | Backend::CoreMl => Placement::Host,
            Backend::Cuda | Backend::TensorRt => Placement::Device,
        }
    }

    /// CUDA graphs are used on GPU backends (for models on the CUDA provider) unless `WAJH_CUDA_GRAPH=0`.
    fn cuda_graphs(&self) -> bool {
        matches!(self.backend, Backend::Cuda | Backend::TensorRt) && std::env::var("WAJH_CUDA_GRAPH").map_or(true, |v| v != "0")
    }

    /// A session for the model at `path`, or for `bytes` (a patched copy of it) when given.
    fn session(&self, path: &Path, bytes: Option<&[u8]>, options: &SessionOptions<'_>, dimensions: &[(&str, i64)]) -> ort::Result<Session> {
        // CUDA runs whole graphs on the GPU; CoreML may leave nodes on the CPU, which want threads.
        let gpu = matches!(self.backend, Backend::Cuda | Backend::TensorRt);
        let mut builder = Session::builder()?;
        for &(name, size) in dimensions {
            builder = builder.with_dimension_override(name, size).map_err(ort::Error::<()>::from)?;
        }
        // `WAJH_ORT_PROFILE=<dir>` writes one ONNX Runtime trace per model (chrome://tracing).
        if let Some(dir) = std::env::var_os("WAJH_ORT_PROFILE") {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            builder = builder.with_profiling(Path::new(&dir).join(stem)).map_err(ort::Error::<()>::from)?;
        }
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(ort::Error::<()>::from)?
            .with_intra_threads(if gpu { 1 } else { self.cpu_threads })
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers(self.execution_providers(options))
            .map_err(ort::Error::<()>::from)?;
        match bytes {
            Some(bytes) => builder.commit_from_memory(bytes),
            None => builder.commit_from_file(path),
        }
    }

    fn execution_providers(&self, options: &SessionOptions<'_>) -> Vec<ExecutionProviderDispatch> {
        let cuda_graph = options.cuda_graph;
        let cache = data_dir().join("trt_cache");
        let tensorrt = || {
            let _ = std::fs::create_dir_all(&cache);
            let cache = cache.to_string_lossy().into_owned();
            TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(&cache)
                .with_engine_cache_prefix(options.cache_prefix)
                .with_timing_cache(true)
                .with_timing_cache_path(&cache)
                .with_cuda_graph(cuda_graph)
                .build()
        };
        let cuda = || {
            // NHWC layout and conv+bias fusion were both measured slower on inswapper (fp16, sm_89).
            CUDA::default()
                .with_conv_algorithm_search(ConvAlgorithmSearch::Exhaustive)
                .with_conv_max_workspace(true)
                .with_tf32(true)
                .with_cuda_graph(cuda_graph)
                .build()
        };
        let coreml = || {
            let cache = data_dir().join("coreml_cache");
            let _ = std::fs::create_dir_all(&cache);
            CoreML::default()
                .with_model_format(ModelFormat::MLProgram)
                .with_compute_units(ComputeUnits::All)
                .with_static_input_shapes(true)
                .with_model_cache_dir(cache.to_string_lossy())
                .build()
        };
        match self.backend {
            Backend::TensorRt if options.tensorrt => vec![tensorrt(), cuda()],
            Backend::TensorRt | Backend::Cuda => vec![cuda()],
            Backend::CoreMl => vec![coreml()],
            Backend::Cpu => Vec::new(),
        }
    }
}

/// TensorRT engine cache prefix for `asset` with `dimensions` pinned (and a patched graph).
fn cache_prefix(asset: &Asset, dimensions: &[(&str, i64)], patched: bool) -> String {
    let mut prefix = asset.name.to_owned();
    for (name, size) in dimensions {
        prefix.push('_');
        prefix.extend(name.chars().filter(char::is_ascii_alphanumeric));
        prefix.push_str(&size.to_string());
    }
    if patched {
        prefix.push_str("_patched");
    }
    prefix.retain(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    prefix
}

/// ONNX Runtime scales best with one thread per physical core; `WAJH_THREADS` overrides.
fn cpu_threads() -> usize {
    std::env::var("WAJH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (std::thread::available_parallelism().map_or(8, std::num::NonZero::get) / 2).max(1))
}

#[cfg(test)]
mod tests {
    use super::cache_prefix;
    use crate::assets;

    #[test]
    fn engine_cache_prefix_names_the_pinned_shapes() {
        let at = |size| cache_prefix(&assets::DETECTOR, &[("input_H", size)], false);
        assert_ne!(at(640), at(192));
        assert_eq!(at(192), "scrfd_2.5g_inputH192");
        assert_eq!(cache_prefix(&assets::PARSER, &[], true), "bisenet_resnet_34_patched");
    }
}
