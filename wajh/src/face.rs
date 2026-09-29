//! Face models: SCRFD detector, ArcFace recognizer, face swappers and the GPEN enhancer.

use crate::engine::{Engine, Model};
use crate::geom::{Affine, Landmarks5, Template};
use crate::imgops::{self, Frame};
use crate::models::{self, SwapModel};
use crate::onnx_proto;
use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct Face {
    pub bbox: [f32; 4],
    pub score: f32,
    pub kps: Landmarks5,
}

impl Face {
    pub fn size(&self) -> f32 {
        ((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])).max(0.0).sqrt()
    }
}

pub struct Detector {
    model: Model,
    pub input_size: usize,
    pub threshold: f32,
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter).max(1e-6)
}

impl Detector {
    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&models::ensure(&models::DETECTOR)?)?, input_size: 640, threshold: 0.5 })
    }

    /// Faces sorted largest first.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        let s = self.input_size;
        let (input, ratio) = imgops::detector_input(frame, s);
        let out = self.model.run(vec![("input", vec![1, 3, s as i64, s as i64], input)])?;
        if out.len() < 9 {
            bail!("unexpected SCRFD outputs");
        }
        let mut faces = Vec::new();
        for (idx, stride) in [8usize, 16, 32].into_iter().enumerate() {
            let (scores, boxes, kps) = (&out[idx].data, &out[idx + 3].data, &out[idx + 6].data);
            let fw = s / stride;
            let st = stride as f32;
            for (i, &score) in scores.iter().enumerate() {
                if score < self.threshold {
                    continue;
                }
                let cell = i / 2;
                let (ax, ay) = (((cell % fw) * stride) as f32, ((cell / fw) * stride) as f32);
                let b = &boxes[i * 4..i * 4 + 4];
                let bbox = [(ax - b[0] * st) * ratio, (ay - b[1] * st) * ratio, (ax + b[2] * st) * ratio, (ay + b[3] * st) * ratio];
                let k = &kps[i * 10..i * 10 + 10];
                let kps: Landmarks5 = std::array::from_fn(|j| [(ax + k[j * 2] * st) * ratio, (ay + k[j * 2 + 1] * st) * ratio]);
                faces.push(Face { bbox, score, kps });
            }
        }
        faces.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut keep: Vec<Face> = Vec::new();
        for f in faces {
            if keep.iter().all(|k| iou(&k.bbox, &f.bbox) < 0.4) {
                keep.push(f);
            }
        }
        keep.sort_by(|a, b| b.size().total_cmp(&a.size()));
        Ok(keep)
    }
}

pub struct Recognizer {
    model: Model,
}

impl Recognizer {
    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&models::ensure(&models::RECOGNIZER)?)? })
    }

    /// Raw (un-normalized) 512-d ArcFace embedding.
    pub fn embed(&mut self, frame: &Frame, kps: &Landmarks5) -> Result<Vec<f32>> {
        let m = Template::Arcface112V2.align(kps, 112);
        let crop = imgops::warp_crop(frame, &m, 112);
        let input = imgops::to_nchw(&crop, 112, true, 0.5, 0.5);
        let out = self.model.run(vec![("input", vec![1, 3, 112, 112], input)])?;
        Ok(out.into_iter().next().context("no embedding")?.data)
    }
}

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

pub struct Swapper {
    model: Model,
    pub kind: SwapModel,
    pub size: usize,
    emap: Option<Vec<f32>>,
}

/// inswapper's 512x512 embedding map, cached next to the model after the first extraction.
fn load_emap(model_path: &std::path::Path) -> Result<Vec<f32>> {
    let cache = model_path.with_extension("emap");
    if let Ok(bytes) = std::fs::read(&cache) {
        if bytes.len() == 512 * 512 * 4 {
            return Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
        }
    }
    let bytes = std::fs::read(model_path)?;
    let init = onnx_proto::last_initializer(&bytes)?;
    if init.data.len() != 512 * 512 {
        bail!("unexpected emap shape {:?}", init.dims);
    }
    let raw: Vec<u8> = init.data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let _ = std::fs::write(&cache, raw);
    Ok(init.data)
}

impl Swapper {
    pub fn new(engine: &Engine, kind: SwapModel) -> Result<Self> {
        let path = models::ensure(&kind.file())?;
        let model = engine.load(&path)?;
        let (size, emap) = match kind {
            SwapModel::Inswapper => (128, Some(load_emap(&path)?)),
            _ => (256, None),
        };
        Ok(Self { model, kind, size, emap })
    }

    /// Convert a raw ArcFace embedding into this model's `source` input.
    pub fn prepare_source(&self, embedding: &[f32]) -> Vec<f32> {
        match &self.emap {
            Some(emap) => {
                let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let mut out = vec![0f32; 512];
                for (i, e) in embedding.iter().enumerate() {
                    let row = &emap[i * 512..(i + 1) * 512];
                    for (o, w) in out.iter_mut().zip(row) {
                        *o += e * w;
                    }
                }
                out.iter_mut().for_each(|o| *o /= norm);
                out
            }
            None => l2_normalize(embedding),
        }
    }

    fn norm(&self) -> (f32, f32) {
        match self.kind {
            SwapModel::Inswapper => (0.0, 1.0),
            _ => (0.5, 0.5),
        }
    }

    /// Returns the swapped aligned crop and the frame->crop transform.
    pub fn swap(&mut self, frame: &Frame, kps: &Landmarks5, source: &[f32]) -> Result<(Vec<u8>, Affine)> {
        let s = self.size;
        let m = Template::Arcface128.align(kps, s);
        let crop = imgops::warp_crop(frame, &m, s);
        let (mean, std) = self.norm();
        let target = imgops::to_nchw(&crop, s, true, mean, std);
        let out = self.model.run(vec![("target", vec![1, 3, s as i64, s as i64], target), ("source", vec![1, 512], source.to_vec())])?;
        let out = out.into_iter().next().context("no swap output")?;
        Ok((imgops::from_nchw(&out.data, s, true, mean, std), m))
    }
}

pub struct Enhancer {
    model: Model,
}

impl Enhancer {
    pub const SIZE: usize = 256;

    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&models::ensure(&models::ENHANCER)?)? })
    }

    pub fn enhance(&mut self, frame: &Frame, kps: &Landmarks5) -> Result<(Vec<u8>, Affine)> {
        let s = Self::SIZE;
        let m = Template::Arcface128.align(kps, s);
        let crop = imgops::warp_crop(frame, &m, s);
        let input = imgops::to_nchw(&crop, s, true, 0.5, 0.5);
        let name = self.model.inputs.first().cloned().unwrap_or_else(|| "input".into());
        let out = self.model.run(vec![(name.as_str(), vec![1, 3, s as i64, s as i64], input)])?;
        let out = out.into_iter().next().context("no enhancer output")?;
        Ok((imgops::from_nchw(&out.data, s, true, 0.5, 0.5), m))
    }
}
