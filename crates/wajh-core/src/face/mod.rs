//! Face models: detection, identity embedding, swapping and restoration.

mod detector;
mod enhancer;
mod recognizer;
mod swapper;

pub use detector::Detector;
pub use enhancer::Enhancer;
pub use recognizer::{Embedding, Recognizer};
pub use swapper::Swapper;

use crate::geometry::{Affine, Landmarks5};

/// A detected face in frame coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    pub score: f32,
    /// Eyes, nose tip, mouth corners.
    pub landmarks: Landmarks5,
}

impl Face {
    /// Geometric mean of the box sides; a resolution-independent notion of face scale.
    #[must_use]
    pub fn size(&self) -> f32 {
        ((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])).max(0.0).sqrt()
    }
}

/// A model output aligned to the face: a square BGR crop plus the frame → crop transform.
#[derive(Clone, Debug)]
pub struct AlignedCrop {
    pub pixels: Vec<u8>,
    pub size: usize,
    pub to_crop: Affine,
}
