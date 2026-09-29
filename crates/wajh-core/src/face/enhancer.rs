use super::AlignedCrop;
use crate::assets;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, from_nchw, to_nchw_into, warp_crop_into};
use crate::runtime::{Engine, StaticModel};
use crate::Result;

/// GPEN-BFR 256 face restoration.
pub struct Enhancer {
    model: StaticModel,
    /// Reused aligned-crop buffer.
    crop: Vec<u8>,
}

impl Enhancer {
    pub const SIZE: usize = 256;

    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load_static(&assets::ENHANCER)?, crop: vec![0; Self::SIZE * Self::SIZE * Frame::CHANNELS] })
    }

    pub fn enhance(&mut self, frame: &Frame, landmarks: &Landmarks5) -> Result<AlignedCrop> {
        let size = Self::SIZE;
        let norm = Normalization::SIGNED_RGB;
        let to_crop = Template::Arcface128.align(landmarks, size);
        warp_crop_into(frame, &to_crop, size, &mut self.crop);
        to_nchw_into(&self.crop, size, norm, self.model.input_mut(0));
        self.model.run()?;
        Ok(AlignedCrop { pixels: from_nchw(self.model.output(0), size, norm), size, to_crop })
    }
}
