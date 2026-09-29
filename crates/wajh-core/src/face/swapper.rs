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
}

impl Swapper {
    pub fn new(engine: &Engine, kind: SwapModel) -> Result<Self> {
        let asset = kind.asset();
        let model = engine.load(&asset)?;
        let embedding_map = match kind {
            SwapModel::Inswapper => Some(load_embedding_map(&asset.path())?),
            _ => None,
        };
        Ok(Self { model, kind, embedding_map })
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
        for (value, row) in identity.0.iter().zip(map.chunks_exact(EMBEDDING_LEN)) {
            out.iter_mut().zip(row).for_each(|(o, w)| *o = value.mul_add(*w, *o));
        }
        out.iter_mut().for_each(|o| *o /= norm);
        out
    }

    const fn normalization(&self) -> Normalization {
        match self.kind {
            SwapModel::Inswapper => Normalization::UNIT_RGB,
            _ => Normalization::SIGNED_RGB,
        }
    }

    pub fn swap(&mut self, frame: &Frame, landmarks: &Landmarks5, condition: &[f32]) -> Result<AlignedCrop> {
        let size = self.kind.resolution();
        let norm = self.normalization();
        let to_crop = Template::Arcface128.align(landmarks, size);
        let target = to_nchw(&warp_crop(frame, &to_crop, size), size, norm);
        let output = self.model.run_single(vec![Input::image("target", size, target), Input::vector("source", condition.to_vec())])?;
        Ok(AlignedCrop { pixels: from_nchw(&output, size, norm), size, to_crop })
    }
}

/// Read inswapper's embedding map from the ONNX file, caching it beside the model.
fn load_embedding_map(model_path: &Path) -> Result<Vec<f32>> {
    const LEN: usize = EMBEDDING_LEN * EMBEDDING_LEN;
    let cache = model_path.with_extension("emap");
    if let Ok(bytes) = std::fs::read(&cache)
        && bytes.len() == LEN * 4
    {
        return Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    let initializer = proto::last_initializer(&std::fs::read(model_path)?)?;
    if initializer.data.len() != LEN {
        return Err(Error::InvalidOnnx(format!("embedding map has shape {:?}", initializer.dims)));
    }
    let bytes: Vec<u8> = initializer.data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&cache, bytes)?;
    Ok(initializer.data)
}
