use crate::assets;
use crate::runtime::{Engine, Input, Model};
use crate::{Error, Result};
use rayon::prelude::*;

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

/// `BiSeNet` face parser: per-pixel class labels for an aligned face crop.
pub struct Parser {
    model: Model,
    input_name: String,
}

impl Parser {
    /// Side length of the (`Arcface128`-aligned) crop the parser works on.
    pub const SIZE: usize = 512;
    const CLASSES: usize = 19;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    pub fn new(engine: &Engine) -> Result<Self> {
        let model = engine.load(&assets::PARSER)?;
        let input_name = model.input_names().first().map_or_else(|| "input".to_owned(), |n| (*n).to_owned());
        Ok(Self { model, input_name })
    }

    /// `1.0` where a pixel of the `SIZE`² BGR crop belongs to `regions`, else `0.0`.
    pub fn regions(&mut self, crop: &[u8], regions: FaceRegions) -> Result<Vec<f32>> {
        let plane = Self::SIZE * Self::SIZE;
        let mut input = vec![0f32; plane * 3];
        for c in 0..3 {
            let (scale, bias) = (1.0 / (255.0 * Self::STD[c]), -Self::MEAN[c] / Self::STD[c]);
            // Model channel c is RGB; the crop is BGR.
            for (dst, px) in input[c * plane..(c + 1) * plane].iter_mut().zip(crop.as_chunks::<3>().0) {
                *dst = f32::from(px[2 - c]).mul_add(scale, bias);
            }
        }
        let size = Self::SIZE as i64;
        let logits = self.model.run_single(vec![Input { name: &self.input_name, shape: vec![1, 3, size, size], data: input }])?;
        if logits.len() != plane * Self::CLASSES {
            return Err(Error::UnexpectedOutput { model: assets::PARSER.name, detail: format!("{} values", logits.len()) });
        }
        let mut mask = vec![0f32; plane];
        mask.par_chunks_mut(Self::SIZE).enumerate().for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                let i = y * Self::SIZE + x;
                let class =
                    (0..Self::CLASSES).map(|c| logits[c * plane + i]).enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(c, _)| c);
                *value = if regions.contains_class(class as u8) { 1.0 } else { 0.0 };
            }
        });
        Ok(mask)
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
}
