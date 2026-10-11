use crate::assets::Asset;
use crate::runtime::proto::{DataType, GraphPatch, Node, TensorInfo};
use crate::runtime::{Engine, StaticModel, StaticShapes};
use crate::{Error, Result};

/// Parts of the face the `BiSeNet` parser can separate (`CelebAMask-HQ` classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum FaceRegion {
    Skin,
    Eyebrows,
    Eyes,
    Glasses,
    Nose,
    /// Inside of the mouth (teeth, tongue).
    Mouth,
    Lips,
}

impl FaceRegion {
    pub const ALL: [Self; 7] = [Self::Skin, Self::Eyebrows, Self::Eyes, Self::Glasses, Self::Nose, Self::Mouth, Self::Lips];

    /// Parser class indices belonging to this region.
    #[must_use]
    pub const fn classes(self) -> &'static [u8] {
        match self {
            Self::Skin => &[1],
            Self::Eyebrows => &[2, 3],
            Self::Eyes => &[4, 5],
            Self::Glasses => &[6],
            Self::Nose => &[10],
            Self::Mouth => &[11],
            Self::Lips => &[12, 13],
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Skin => "Skin",
            Self::Eyebrows => "Eyebrows",
            Self::Eyes => "Eyes",
            Self::Glasses => "Glasses",
            Self::Nose => "Nose",
            Self::Mouth => "Mouth",
            Self::Lips => "Lips",
        }
    }
}

/// A set of [`FaceRegion`]s, stored as a bitmask over parser classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceRegions(u32);

impl FaceRegions {
    pub const NONE: Self = Self(0);
    /// Everything that belongs to the face itself (no hair, ears, neck or background).
    pub const ALL: Self = Self::from_regions(&FaceRegion::ALL);
    /// The mouth and lips: what "keep the original mouth" preserves.
    pub const MOUTH: Self = Self::from_regions(&[FaceRegion::Mouth, FaceRegion::Lips]);

    #[must_use]
    pub const fn from_regions(regions: &[FaceRegion]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < regions.len() {
            let classes = regions[i].classes();
            let mut j = 0;
            while j < classes.len() {
                bits |= 1 << classes[j];
                j += 1;
            }
            i += 1;
        }
        Self(bits)
    }

    #[must_use]
    pub const fn contains(self, region: FaceRegion) -> bool {
        let bits = Self::from_regions(&[region]).0;
        self.0 & bits == bits
    }

    #[must_use]
    pub const fn contains_class(self, class: u8) -> bool {
        class < 32 && self.0 & (1 << class) != 0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub const fn with(self, region: FaceRegion, on: bool) -> Self {
        let bits = Self::from_regions(&[region]).0;
        Self(if on { self.0 | bits } else { self.0 & !bits })
    }

    pub fn iter(self) -> impl Iterator<Item = FaceRegion> {
        FaceRegion::ALL.into_iter().filter(move |&r| self.contains(r))
    }
}

impl Default for FaceRegions {
    fn default() -> Self {
        Self::ALL
    }
}

impl FromIterator<FaceRegion> for FaceRegions {
    fn from_iter<I: IntoIterator<Item = FaceRegion>>(iter: I) -> Self {
        iter.into_iter().fold(Self::NONE, |set, r| set.with(r, true))
    }
}

/// Name of the logits output in `FaceFusion`'s `BiSeNet` exports (the other two are training-time
/// auxiliary heads, pruned away).
const LOGITS: &str = "output";
/// Value names added by the label-map patch.
const LABELS_I64: &str = "wessh_labels_i64";
const LABELS: &str = "wessh_labels";
/// `Cast.to` for `float`.
const CAST_TO_FLOAT: i64 = DataType::Float as i64;

/// `BiSeNet` face parser: per-pixel class labels for an aligned face crop.
///
/// The network is patched on load so it computes the label map itself: the auxiliary heads are
/// pruned, an `ArgMax` over the 19 class logits is appended, and only the `size`² label map
/// comes back from the GPU instead of three `19×512×512` logit tensors. Its fixed-kernel
/// global pools become `GlobalAveragePool`, identical at the trained 512 px and valid at any
/// other input size.
pub struct Parser {
    model: StaticModel,
    size: usize,
}

impl Parser {
    /// Side length the parser was trained at.
    pub const NATIVE_SIZE: usize = 512;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// Load `asset` (a `BiSeNet` export) running on `size`² crops.
    pub fn new(engine: &Engine, asset: &Asset, size: usize) -> Result<Self> {
        let side = size as i64;
        let input_dims = [1, 3, side, side];
        let label_dims = [1, side, side];
        let patch = GraphPatch {
            inputs: &[TensorInfo { name: "input", data_type: DataType::Float, dims: &input_dims }],
            retype: &[("AveragePool", "GlobalAveragePool")],
            unpad_conv_transpose: false,
            nodes: &[
                Node {
                    op_type: "ArgMax",
                    name: "wessh_argmax",
                    inputs: &[LOGITS],
                    outputs: &[LABELS_I64],
                    attributes: &[("axis", 1), ("keepdims", 0)],
                },
                Node {
                    op_type: "Cast",
                    name: "wessh_cast",
                    inputs: &[LABELS_I64],
                    outputs: &[LABELS],
                    attributes: &[("to", CAST_TO_FLOAT)],
                },
            ],
            outputs: &[TensorInfo { name: LABELS, data_type: DataType::Float, dims: &label_dims }],
        };
        let outputs = [vec![1, size, size]];
        let model = engine.load_static_with(asset, &StaticShapes { patch: Some(&patch), outputs: &outputs, ..StaticShapes::default() })?;
        if model.output(0).len() != size * size {
            return Err(Error::UnexpectedOutput { model: asset.name, detail: format!("{} labels", model.output(0).len()) });
        }
        Ok(Self { model, size })
    }

    /// Side of the crops [`Parser::parse`] takes.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Class label of every pixel of a `size`² BGR crop (as `f32`).
    pub fn parse(&mut self, crop: &[u8]) -> Result<&[f32]> {
        let plane = self.size * self.size;
        let input = self.model.input_mut(0);
        for c in 0..3 {
            let (scale, bias) = (1.0 / (255.0 * Self::STD[c]), -Self::MEAN[c] / Self::STD[c]);
            // Model channel c is RGB; the crop is BGR.
            for (dst, px) in input[c * plane..(c + 1) * plane].iter_mut().zip(crop.as_chunks::<3>().0) {
                *dst = f32::from(px[2 - c]).mul_add(scale, bias);
            }
        }
        self.model.run()?;
        Ok(self.model.output(0))
    }
}

/// `1.0` where a label belongs to `regions`, else `0.0`.
pub fn select_regions(labels: &[f32], regions: FaceRegions, out: &mut [f32]) {
    for (o, &label) in out.iter_mut().zip(labels) {
        *o = f32::from(u8::from(regions.contains_class(label as u8)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_sets() {
        assert!(FaceRegions::ALL.contains(FaceRegion::Glasses));
        assert!(!FaceRegions::ALL.contains_class(0) && !FaceRegions::ALL.contains_class(17)); // background, hair
        let mouth: FaceRegions = [FaceRegion::Mouth, FaceRegion::Lips].into_iter().collect();
        assert_eq!(mouth, FaceRegions::MOUTH);
        assert!(mouth.contains_class(12) && mouth.contains_class(13) && !mouth.contains_class(1));
        assert_eq!(FaceRegions::ALL.with(FaceRegion::Skin, false).iter().count(), 6);
        assert!(FaceRegions::NONE.is_empty());
    }

    #[test]
    fn selects_labels() {
        let mut out = [9.0; 4];
        select_regions(&[0.0, 12.0, 11.0, 1.0], FaceRegions::MOUTH, &mut out);
        assert_eq!(out, [0.0, 1.0, 1.0, 0.0]);
    }
}
