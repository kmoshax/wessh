use super::{AlignedCrop, Embedding};
use crate::assets::SwapModel;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, from_nchw, to_nchw, warp_crop};
use crate::runtime::{Engine, Input, Model, proto};
use crate::{Error, Result};
use std::path::Path;

const EMBEDDING_LEN: usize = 512;

/// Identity-conditioned face swap network.
pub struct Swapper {
    model: Model,
    kind: SwapModel,
    /// inswapper's learned 512×512 projection of the ArcFace embedding.
    embedding_map: Option<Vec<f32>>,
    /// Whether the model accepts batches; `None` until a batched run has been tried.
    batching: Option<bool>,
}

impl Swapper {
    pub fn new(engine: &Engine, kind: SwapModel) -> Result<Self> {
        let asset = kind.asset();
        let model = engine.load(&asset)?;
        let embedding_map = match kind {
            SwapModel::Inswapper => Some(load_embedding_map(&asset.path())?),
            _ => None,
        };
        Ok(Self { model, kind, embedding_map, batching: None })
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
        if crops.len() > 1 && self.batching != Some(false) {
            match self.run(crops, condition) {
                Ok(out) => {
                    self.batching = Some(true);
                    return Ok(out);
                }
                // A fixed batch dimension rejects the first batched run; go one by one from now on.
                Err(_) if self.batching.is_none() => self.batching = Some(false),
                Err(e) => return Err(e),
            }
        }
        crops.iter().map(|crop| Ok(self.run(std::slice::from_ref(crop), condition)?.swap_remove(0))).collect()
    }

    fn run(&mut self, crops: &[Vec<u8>], condition: &[f32]) -> Result<Vec<Vec<u8>>> {
        let size = self.kind.resolution();
        let norm = self.normalization();
        let n = crops.len();
        let target: Vec<f32> = crops.iter().flat_map(|crop| to_nchw(crop, size, norm)).collect();
        let output = self.model.run_single(vec![
            Input { name: "target", shape: vec![n as i64, 3, size as i64, size as i64], data: target },
            Input { name: "source", shape: vec![n as i64, condition.len() as i64], data: condition.repeat(n) },
        ])?;
        let len = 3 * size * size;
        if output.len() != n * len {
            return Err(Error::UnexpectedOutput {
                model: self.kind.asset().name,
                detail: format!("{} values for {n} crops", output.len()),
            });
        }
        Ok(output.chunks_exact(len).map(|tensor| from_nchw(tensor, size, norm)).collect())
    }
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
