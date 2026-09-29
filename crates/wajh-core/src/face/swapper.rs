use super::{AlignedCrop, Embedding};
use crate::assets::SwapModel;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, from_nchw, to_nchw, to_nchw_into, warp_crop};
use crate::runtime::{Engine, Input, Model, StaticModel, StaticShapes, proto};
use crate::{Error, Result};
use std::path::Path;

const EMBEDDING_LEN: usize = 512;

/// Identity-conditioned face swap network.
pub struct Swapper {
    network: Network,
    kind: SwapModel,
    /// inswapper's learned 512×512 projection of the ArcFace embedding.
    embedding_map: Option<Vec<f32>>,
}

/// How the network is run.
enum Network {
    /// Fixed batch of one: pre-bound buffers (and a CUDA graph on GPU), one run per crop.
    Static(Box<BoundSwap>),
    /// Symbolic shapes: fresh tensors per run, crops batched when the model allows it.
    Dynamic {
        model: Model,
        /// Whether the model accepts batches; `None` until a batched run has been tried.
        batching: Option<bool>,
    },
}

/// A fixed-shape swap network with its inputs bound once.
struct BoundSwap {
    model: StaticModel,
    target: usize,
    source: usize,
    /// Condition currently in the `source` input; the identity rarely changes, so it is only
    /// uploaded when it does.
    bound_condition: Option<Vec<f32>>,
}

impl Swapper {
    pub fn new(engine: &Engine, kind: SwapModel) -> Result<Self> {
        let asset = kind.asset();
        // TensorRT's fp16 build of inswapper (already an fp16 export) produces a washed-out,
        // noisy face; fp32 matches CUDA and is still faster.
        let shapes = StaticShapes { tensorrt_fp32: kind == SwapModel::Inswapper, ..StaticShapes::default() };
        let network = match engine.load_static_with(&asset, &shapes) {
            Ok(model) => {
                let index = |name: &str| {
                    model
                        .input_index(name)
                        .ok_or_else(|| Error::UnexpectedOutput { model: asset.name, detail: format!("no `{name}` input") })
                };
                let (target, source) = (index("target")?, index("source")?);
                Network::Static(Box::new(BoundSwap { model, target, source, bound_condition: None }))
            }
            Err(Error::DynamicShape { .. }) => Network::Dynamic { model: engine.load(&asset)?, batching: None },
            Err(e) => return Err(e),
        };
        let embedding_map = match kind {
            SwapModel::Inswapper => Some(load_embedding_map(&asset.path())?),
            _ => None,
        };
        Ok(Self { network, kind, embedding_map })
    }

    #[must_use]
    pub const fn kind(&self) -> SwapModel {
        self.kind
    }

    /// Convert an identity embedding into this network's `source` input. Compute once per identity.
    #[must_use]
    pub fn condition(&self, identity: &Embedding) -> Vec<f32> {
        let Some(map) = &self.embedding_map else {
            return identity.normalized();
        };
        let norm = identity.norm();
        let mut out = vec![0f32; EMBEDDING_LEN];
        for (value, row) in identity.0.iter().zip(map.as_chunks::<EMBEDDING_LEN>().0) {
            out.iter_mut().zip(row).for_each(|(o, w)| *o = value.mul_add(*w, *o));
        }
        for o in &mut out {
            *o /= norm;
        }
        out
    }

    const fn normalization(&self) -> Normalization {
        match self.kind {
            SwapModel::Inswapper => Normalization::UNIT_RGB,
            _ => Normalization::SIGNED_RGB,
        }
    }

    /// Swap the face at `landmarks` at the network's native resolution.
    pub fn swap(&mut self, frame: &Frame, landmarks: &Landmarks5, condition: &[f32]) -> Result<AlignedCrop> {
        let size = self.kind.resolution();
        let to_crop = Template::Arcface128.align(landmarks, size);
        let pixels = self.swap_crops(&[warp_crop(frame, &to_crop, size)], condition)?.swap_remove(0);
        Ok(AlignedCrop { pixels, size, to_crop })
    }

    /// Swap already-aligned native-resolution BGR crops, as one batch when the model allows it.
    pub fn swap_crops(&mut self, crops: &[Vec<u8>], condition: &[f32]) -> Result<Vec<Vec<u8>>> {
        let (size, norm) = (self.kind.resolution(), self.normalization());
        match &mut self.network {
            Network::Static(bound) => crops.iter().map(|crop| bound.run(crop, condition, size, norm)).collect(),
            Network::Dynamic { model, batching } => {
                if crops.len() > 1 && *batching != Some(false) {
                    match run_dynamic(model, crops, condition, size, norm) {
                        Ok(out) => {
                            *batching = Some(true);
                            return Ok(out);
                        }
                        // A fixed batch dimension rejects the first batched run; go one by one from now on.
                        Err(_) if batching.is_none() => *batching = Some(false),
                        Err(e) => return Err(e),
                    }
                }
                crops.iter().map(|crop| Ok(run_dynamic(model, std::slice::from_ref(crop), condition, size, norm)?.swap_remove(0))).collect()
            }
        }
    }
}

impl BoundSwap {
    fn run(&mut self, crop: &[u8], condition: &[f32], size: usize, norm: Normalization) -> Result<Vec<u8>> {
        to_nchw_into(crop, size, norm, self.model.input_mut(self.target));
        if self.bound_condition.as_deref() != Some(condition) {
            self.model.input_mut(self.source).copy_from_slice(condition);
            self.bound_condition = Some(condition.to_vec());
        }
        self.model.run()?;
        Ok(from_nchw(self.model.output(0), size, norm))
    }
}

/// One run of a dynamically shaped swap network over `crops` as a batch.
fn run_dynamic(model: &mut Model, crops: &[Vec<u8>], condition: &[f32], size: usize, norm: Normalization) -> Result<Vec<Vec<u8>>> {
    let n = crops.len();
    let target: Vec<f32> = crops.iter().flat_map(|crop| to_nchw(crop, size, norm)).collect();
    let output = model.run_single(vec![
        Input { name: "target", shape: vec![n as i64, 3, size as i64, size as i64], data: target },
        Input { name: "source", shape: vec![n as i64, condition.len() as i64], data: condition.repeat(n) },
    ])?;
    let len = 3 * size * size;
    if output.len() != n * len {
        return Err(Error::UnexpectedOutput { model: "swapper", detail: format!("{} values for {n} crops", output.len()) });
    }
    Ok(output.chunks_exact(len).map(|tensor| from_nchw(tensor, size, norm)).collect())
}

/// Read inswapper's embedding map from the ONNX file, caching it beside the model.
fn load_embedding_map(model_path: &Path) -> Result<Vec<f32>> {
    const LEN: usize = EMBEDDING_LEN * EMBEDDING_LEN;
    let cache = model_path.with_extension("emap");
    if let Ok(bytes) = std::fs::read(&cache)
        && bytes.len() == LEN * 4
    {
        return Ok(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect());
    }
    let initializer = proto::last_initializer(&std::fs::read(model_path)?)?;
    if initializer.data.len() != LEN {
        return Err(Error::InvalidOnnx(format!("embedding map has shape {:?}", initializer.dims)));
    }
    let bytes: Vec<u8> = initializer.data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&cache, bytes)?;
    Ok(initializer.data)
}
