//! ONNX Runtime setup: dynamic loading, execution-provider selection and a thin model wrapper.

use crate::models;
use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    /// TensorRT if available, then CUDA, then CPU.
    Auto,
    Tensorrt,
    Cuda,
    Cpu,
}

#[derive(Clone)]
pub struct Engine {
    pub backend: &'static str,
    use_trt: bool,
    use_cuda: bool,
    trt_cache: PathBuf,
}

fn provider_lib(dir: &Path, name: &str) -> bool {
    let file = if cfg!(target_os = "windows") {
        format!("onnxruntime_providers_{name}.dll")
    } else {
        format!("libonnxruntime_providers_{name}.so")
    };
    dir.join(file).exists()
}

impl Engine {
    pub fn init(ort_lib: Option<&Path>, device: Device) -> Result<Self> {
        let lib = models::find_onnxruntime(ort_lib)?;
        ort::init_from(&lib).map_err(|e| anyhow!("loading {}: {e}", lib.display()))?.with_name("wajh").commit();
        let dir = lib.parent().unwrap_or(Path::new("."));
        let (has_cuda, has_trt) = (provider_lib(dir, "cuda"), provider_lib(dir, "tensorrt"));
        let use_trt = has_trt && matches!(device, Device::Auto | Device::Tensorrt);
        let use_cuda = has_cuda && matches!(device, Device::Auto | Device::Tensorrt | Device::Cuda);
        if matches!(device, Device::Tensorrt) && !has_trt {
            log_warn("TensorRT provider library not found next to ONNX Runtime; falling back");
        }
        if matches!(device, Device::Cuda | Device::Tensorrt) && !has_cuda {
            log_warn("CUDA provider library not found next to ONNX Runtime; running on CPU");
        }
        let backend = if use_trt {
            "TensorRT"
        } else if use_cuda {
            "CUDA"
        } else {
            "CPU"
        };
        let trt_cache = models::data_dir().join("trt_cache");
        Ok(Self { backend, use_trt, use_cuda, trt_cache })
    }

    pub fn load(&self, path: &Path) -> Result<Model> {
        let mut eps = Vec::new();
        if self.use_trt {
            std::fs::create_dir_all(&self.trt_cache)?;
            let cache = self.trt_cache.to_string_lossy().to_string();
            eps.push(
                ort::ep::TensorRT::default()
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache)
                    .build(),
            );
        }
        if self.use_cuda {
            eps.push(
                ort::ep::CUDA::default()
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
                    .with_conv_max_workspace(true)
                    .with_tf32(true)
                    .build(),
            );
        }
        let threads = std::env::var("WAJH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) / 2).max(1));
        let session = Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(if self.use_cuda { 1 } else { threads })
            .map_err(|e| anyhow!("{e}"))?
            .with_execution_providers(eps)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(path)
            .with_context(|| format!("loading model {}", path.display()))?;
        let inputs = session.inputs().iter().map(|i| i.name().to_string()).collect();
        Ok(Model { session, inputs })
    }
}

fn log_warn(msg: &str) {
    eprintln!("warning: {msg}");
}

pub struct Output {
    pub data: Vec<f32>,
}

pub struct Model {
    session: Session,
    pub inputs: Vec<String>,
}

impl Model {
    /// Run with named f32 inputs `(name, shape, data)` and return all outputs copied out.
    pub fn run(&mut self, inputs: Vec<(&str, Vec<i64>, Vec<f32>)>) -> Result<Vec<Output>> {
        let mut values = Vec::with_capacity(inputs.len());
        for (name, shape, data) in inputs {
            values.push((std::borrow::Cow::from(name.to_string()), ort::session::SessionInputValue::from(Tensor::from_array((shape, data))?)));
        }
        let outputs = self.session.run(values)?;
        let mut out = Vec::with_capacity(outputs.len());
        for (_, v) in outputs.iter() {
            let (_, data) = v.try_extract_tensor::<f32>()?;
            out.push(Output { data: data.to_vec() });
        }
        Ok(out)
    }
}
