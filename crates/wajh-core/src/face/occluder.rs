use crate::assets::OccluderModel;
use crate::runtime::{Engine, Input, Model};
use crate::{Error, Result};

/// `XSeg` face segmentation: `1` where the face is visible, `0` where something covers it.
pub struct Occluder {
    model: Model,
    kind: OccluderModel,
    input_name: String,
}

impl Occluder {
    /// Side length of the (`Arcface128`-aligned) crop `XSeg` works on.
    pub const SIZE: usize = 256;

    pub fn new(engine: &Engine, kind: OccluderModel) -> Result<Self> {
        let model = engine.load(&kind.asset())?;
        let input_name = model.input_names().first().map_or_else(|| "input".to_owned(), |n| (*n).to_owned());
        Ok(Self { model, kind, input_name })
    }

    #[must_use]
    pub const fn kind(&self) -> OccluderModel {
        self.kind
    }

    /// Visibility of each pixel of a `SIZE`² BGR crop, in `[0, 1]`.
    pub fn visible(&mut self, crop: &[u8]) -> Result<Vec<f32>> {
        let size = Self::SIZE as i64;
        // XSeg takes NHWC BGR in [0, 1] — the crop's own layout.
        let data = crop.iter().map(|&v| f32::from(v) / 255.0).collect();
        let input = Input { name: &self.input_name, shape: vec![1, size, size, 3], data };
        let mut mask = self.model.run_single(vec![input])?;
        if mask.len() != Self::SIZE * Self::SIZE {
            return Err(Error::UnexpectedOutput { model: self.kind.asset().name, detail: format!("{} values", mask.len()) });
        }
        for v in &mut mask {
            *v = v.clamp(0.0, 1.0);
        }
        Ok(mask)
    }
}
