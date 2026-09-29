use super::AlignedCrop;
use crate::assets;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, from_nchw, to_nchw, warp_crop};
use crate::runtime::{Engine, Input, Model};
use crate::Result;

/// GPEN-BFR 256 face restoration.
pub struct Enhancer {
    model: Model,
    input_name: String,
}

impl Enhancer {
    pub const SIZE: usize = 256;

    pub fn new(engine: &Engine) -> Result<Self> {
        let model = engine.load(&assets::ENHANCER)?;
        let input_name = model.input_names().first().map_or_else(|| "input".to_owned(), |n| (*n).to_owned());
        Ok(Self { model, input_name })
    }

    pub fn enhance(&mut self, frame: &Frame, landmarks: &Landmarks5) -> Result<AlignedCrop> {
        let size = Self::SIZE;
        let norm = Normalization::SIGNED_RGB;
        let to_crop = Template::Arcface128.align(landmarks, size);
        let input = to_nchw(&warp_crop(frame, &to_crop, size), size, norm);
        let output = self.model.run_single(vec![Input::image(&self.input_name, size, input)])?;
        Ok(AlignedCrop { pixels: from_nchw(&output, size, norm), size, to_crop })
    }
}
