use crate::assets::OccluderModel;
use crate::runtime::proto::{DataType, GraphPatch, TensorInfo};
use crate::runtime::{Engine, StaticModel, StaticShapes};
use crate::{Error, Result};

/// Input of the `XSeg` exports (NHWC, symbolic batch — named differently in every version).
const INPUT: &str = "input";

/// `XSeg` face segmentation: `1` where the face is visible, `0` where something covers it.
pub struct Occluder {
    model: StaticModel,
    kind: OccluderModel,
}

impl Occluder {
    /// Side length of the (`Arcface128`-aligned) crop `XSeg` works on.
    pub const SIZE: usize = 256;

    pub fn new(engine: &Engine, kind: OccluderModel) -> Result<Self> {
        let asset = kind.asset();
        let side = Self::SIZE as i64;
        let input_dims = [1, side, side, 3];
        let outputs = [vec![1, Self::SIZE, Self::SIZE, 1]];
        // XSeg's decoder upsamples with end-padded ConvTransposes, which CUDA would run on the CPU
        // (≈ 20 ms of its ≈ 29 ms); rewritten, the whole network runs on the GPU.
        let patch = GraphPatch {
            inputs: &[TensorInfo { name: INPUT, data_type: DataType::Float, dims: &input_dims }],
            unpad_conv_transpose: true,
            ..GraphPatch::default()
        };
        let model = engine.load_static_with(&asset, &StaticShapes { patch: Some(&patch), outputs: &outputs, ..StaticShapes::default() })?;
        if model.output(0).len() != Self::SIZE * Self::SIZE {
            return Err(Error::UnexpectedOutput { model: asset.name, detail: format!("{} values", model.output(0).len()) });
        }
        Ok(Self { model, kind })
    }

    #[must_use]
    pub const fn kind(&self) -> OccluderModel {
        self.kind
    }

    /// Visibility of each pixel of a `SIZE`² BGR crop (unclamped; the caller clamps to `[0, 1]`).
    pub fn visible(&mut self, crop: &[u8]) -> Result<&[f32]> {
        // XSeg takes NHWC BGR in [0, 1] — the crop's own layout.
        for (dst, &v) in self.model.input_mut(0).iter_mut().zip(crop) {
            *dst = f32::from(v) / 255.0;
        }
        self.model.run()?;
        Ok(self.model.output(0))
    }
}
