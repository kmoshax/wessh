use super::AlignedCrop;
use crate::Result;
use crate::assets;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, Normalization, from_nchw, to_nchw_into, warp_crop_into};
use crate::runtime::{Engine, StaticModel, StaticShapes};

/// GFPGAN 1.4 face restoration: a 512 px FFHQ-aligned face in, a 1024 px restored face out.
pub struct Enhancer {
    model: StaticModel,
    /// Reused aligned-crop buffer.
    crop: Vec<u8>,
}

impl Enhancer {
    /// Side of the aligned input crop.
    pub const INPUT_SIZE: usize = 512;
    /// Side of the restored output.
    pub const OUTPUT_SIZE: usize = 1024;

    pub fn new(engine: &Engine) -> Result<Self> {
        let crop = vec![0; Self::INPUT_SIZE * Self::INPUT_SIZE * Frame::CHANNELS];
        // TensorRT's fp16 build returns a flat purple blob instead of a face; CUDA is correct.
        let model = engine.load_static_with(&assets::ENHANCER, &StaticShapes { skip_tensorrt: true, ..StaticShapes::default() })?;
        Ok(Self { model, crop })
    }

    pub fn enhance(&mut self, frame: &Frame, landmarks: &Landmarks5) -> Result<AlignedCrop> {
        let norm = Normalization::SIGNED_RGB;
        let to_input = Template::Ffhq512.align(landmarks, Self::INPUT_SIZE);
        warp_crop_into(frame, &to_input, Self::INPUT_SIZE, &mut self.crop);
        to_nchw_into(&self.crop, Self::INPUT_SIZE, norm, self.model.input_mut(0));
        self.model.run()?;
        let size = Self::OUTPUT_SIZE;
        Ok(AlignedCrop { pixels: from_nchw(self.model.output(0), size, norm), size, to_crop: Template::Ffhq512.align(landmarks, size) })
    }
}
