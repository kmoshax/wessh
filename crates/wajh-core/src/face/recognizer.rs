use crate::Result;
use crate::assets;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, to_nchw, warp_crop};
use crate::runtime::{Engine, Input, Model};

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

    /// Mean identity of several embeddings of the same person. Directions are averaged after
    /// normalising (so every photo counts equally) and the result keeps the photos' mean
    /// magnitude, since some swappers (`AlphaFace`) take the raw embedding. A few varied photos
    /// give a steadier identity than any single one. Returns `None` for an empty slice.
    #[must_use]
    pub fn average(embeddings: &[Self]) -> Option<Self> {
        let first = embeddings.first()?;
        let n = embeddings.len() as f32;
        let mut direction = vec![0f32; first.0.len()];
        for embedding in embeddings {
            direction.iter_mut().zip(embedding.normalized()).for_each(|(d, v)| *d += v);
        }
        let mean = Self(direction);
        let scale = embeddings.iter().map(Self::norm).sum::<f32>() / n / mean.norm();
        Some(Self(mean.0.into_iter().map(|d| d * scale).collect()))
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

#[cfg(test)]
mod tests {
    use super::Embedding;

    #[test]
    fn average_weighs_every_embedding_equally() {
        let avg = Embedding::average(&[Embedding(vec![10.0, 0.0]), Embedding(vec![0.0, 0.5])]).unwrap();
        assert!((avg.similarity(&Embedding(vec![1.0, 1.0])) - 1.0).abs() < 1e-6, "directions count equally");
        assert!((avg.norm() - 5.25).abs() < 1e-4, "magnitude is the mean of the inputs'");
        assert!(Embedding::average(&[]).is_none());
    }
}
