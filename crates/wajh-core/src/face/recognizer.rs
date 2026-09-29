use crate::assets;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, to_nchw, warp_crop};
use crate::runtime::{Engine, Input, Model};
use crate::Result;

const SIZE: usize = 112;

/// Raw (un-normalised) 512-d ArcFace identity embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.0.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::EPSILON)
    }

    #[must_use]
    pub fn normalized(&self) -> Vec<f32> {
        let norm = self.norm();
        self.0.iter().map(|x| x / norm).collect()
    }

    /// Cosine similarity in `[-1, 1]`.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        self.normalized().iter().zip(other.normalized()).map(|(a, b)| a * b).sum()
    }
}

/// ArcFace (w600k r50) identity encoder.
pub struct Recognizer {
    model: Model,
}

impl Recognizer {
    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&assets::RECOGNIZER)? })
    }

    pub fn embed(&mut self, frame: &Frame, landmarks: &Landmarks5) -> Result<Embedding> {
        let to_crop = Template::Arcface112.align(landmarks, SIZE);
        let crop = warp_crop(frame, &to_crop, SIZE);
        let input = Input::image("input", SIZE, to_nchw(&crop, SIZE, Normalization::SIGNED_RGB));
        Ok(Embedding(self.model.run_single(vec![input])?))
    }
}
