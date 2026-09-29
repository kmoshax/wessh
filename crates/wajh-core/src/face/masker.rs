//! Model-based paste masks (`FaceFusion`'s occlusion and region masks, plus a mouth keep-out).
//!
//! Every mask lives in the Arcface128 template's normalised square, so a mask computed once
//! per face applies to any crop of that template — the swap crop at any pixel-boost size and
//! the enhancer's crop alike. Masks are composed at [`MASK_SIZE`]: they are feathered anyway,
//! and at that size blurring and combining them costs well under a millisecond.

use super::{AlignedCrop, FaceRegions, Occluder, Parser};
use crate::Result;
use crate::assets::OccluderModel;
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, gaussian_blur, resize_crop, resize_plane, warp_crop};
use crate::runtime::Engine;

/// Resolution the model masks are composed at.
pub const MASK_SIZE: usize = 128;
/// Feather radius (Gaussian sigma) as a fraction of the crop — `FaceFusion`'s 5 px at 256.
const FEATHER: f32 = 5.0 / 256.0;

/// Which model masks to apply. Everything off means the plain box mask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaskSpec {
    /// Keep whatever covers the face (hands, hair, glasses, microphones…).
    pub occlusion: Option<OccluderModel>,
    /// Only swap these parts of the face.
    pub regions: Option<FaceRegions>,
    /// Keep the target's own mouth and lips (lip-sync stays natural when talking).
    pub keep_mouth: bool,
}

impl MaskSpec {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.occlusion.is_none() && self.regions.is_none() && !self.keep_mouth
    }

    const fn needs_parser(&self) -> bool {
        self.regions.is_some() || self.keep_mouth
    }
}

/// Lazily loaded occluder and parser models.
#[derive(Default)]
pub struct Masker {
    occluder: Option<Occluder>,
    parser: Option<Parser>,
}

impl Masker {
    /// Load the models `spec` needs (downloading them on first use).
    pub fn prepare(&mut self, engine: &Engine, spec: &MaskSpec) -> Result<()> {
        if let Some(kind) = spec.occlusion
            && self.occluder.as_ref().map(Occluder::kind) != Some(kind)
        {
            self.occluder = None;
            self.occluder = Some(Occluder::new(engine, kind)?);
        }
        if spec.needs_parser() && self.parser.is_none() {
            self.parser = Some(Parser::new(engine)?);
        }
        Ok(())
    }

    /// Intersection of the model masks for one face, `MASK_SIZE`² in the Arcface128 template,
    /// or `None` when `spec` is empty. `swapped` is the swap output for the face at `landmarks`.
    pub fn face_mask(&mut self, frame: &Frame, landmarks: &Landmarks5, swapped: &AlignedCrop, spec: &MaskSpec) -> Result<Option<Vec<f32>>> {
        let mut mask: Option<Vec<f32>> = None;
        let mut add = |m: Vec<f32>| match &mut mask {
            Some(acc) => intersect(acc, &m),
            None => mask = Some(m),
        };
        let sigma = FEATHER * MASK_SIZE as f32;

        if spec.occlusion.is_some()
            && let Some(occluder) = &mut self.occluder
        {
            let crop = aligned(frame, landmarks, Occluder::SIZE);
            let mut visible = resize_plane(&occluder.visible(&crop)?, Occluder::SIZE, MASK_SIZE);
            feather(&mut visible, MASK_SIZE, sigma);
            add(visible);
        }
        if let Some(parser) = &mut self.parser {
            if let Some(regions) = spec.regions {
                // Like FaceFusion, parse the swapped face: its features decide what gets pasted.
                let crop = resize_crop(&swapped.pixels, swapped.size, Parser::SIZE);
                let mut region = resize_plane(&parser.regions(&crop, regions)?, Parser::SIZE, MASK_SIZE);
                feather(&mut region, MASK_SIZE, sigma);
                add(region);
            }
            if spec.keep_mouth {
                // The original mouth: parse the target, grow it a little and cut it out.
                let crop = aligned(frame, landmarks, Parser::SIZE);
                let mut mouth = resize_plane(&parser.regions(&crop, FaceRegions::MOUTH)?, Parser::SIZE, MASK_SIZE);
                keep_out(&mut mouth, MASK_SIZE, sigma * 1.5);
                add(mouth);
            }
        }
        Ok(mask)
    }
}

fn aligned(frame: &Frame, landmarks: &Landmarks5, size: usize) -> Vec<u8> {
    warp_crop(frame, &Template::Arcface128.align(landmarks, size), size)
}

/// Blur a hard mask and remap `[0.5, 1] → [0, 1]`: soft edges that stay inside the region
/// (`FaceFusion`'s `(blur(m).clip(0.5, 1) - 0.5) * 2`).
pub fn feather(mask: &mut [f32], size: usize, sigma: f32) {
    clamp_unit(mask);
    gaussian_blur(mask, size, size, sigma);
    for v in mask {
        *v = ((v.clamp(0.5, 1.0) - 0.5) * 2.0).clamp(0.0, 1.0);
    }
}

/// Turn a hard "keep the original here" region into a paste mask: blurred, grown so the whole
/// region is kept (`[0, 0.5] → [0, 1]`), then inverted.
pub fn keep_out(mask: &mut [f32], size: usize, sigma: f32) {
    clamp_unit(mask);
    gaussian_blur(mask, size, size, sigma);
    for v in mask {
        *v = 1.0 - (*v * 2.0).clamp(0.0, 1.0);
    }
}

fn clamp_unit(mask: &mut [f32]) {
    for v in mask {
        *v = v.clamp(0.0, 1.0);
    }
}

/// `acc = min(acc, other)` — masks combine by intersection.
pub fn intersect(acc: &mut [f32], other: &[f32]) {
    acc.iter_mut().zip(other).for_each(|(a, &b)| *a = a.min(b));
}

/// The final paste mask at `size`: the box mask intersected with a model mask (any template
/// resolution), if there is one.
#[must_use]
pub fn compose(box_mask: &[f32], size: usize, model_mask: Option<(&[f32], usize)>) -> Vec<f32> {
    let mut mask = box_mask.to_vec();
    if let Some((model_mask, model_size)) = model_mask {
        intersect(&mut mask, &resize_plane(model_mask, model_size, size));
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(size: usize, radius: f32) -> Vec<f32> {
        let c = size as f32 / 2.0;
        (0..size * size).map(|i| f32::from(((i % size) as f32 - c).hypot((i / size) as f32 - c) < radius)).collect()
    }

    #[test]
    fn composition_is_the_intersection() {
        let a = [0.0, 0.5, 1.0, 1.0];
        let b = [1.0, 1.0, 0.25, 1.0];
        assert_eq!(compose(&a, 2, Some((&b, 2))), [0.0, 0.5, 0.25, 1.0]);
        assert_eq!(compose(&a, 2, None), a);
        // A coarser model mask is resampled to the paste size first.
        let upsampled = compose(&[1.0; 16], 4, Some((&[0.0, 1.0, 0.0, 1.0], 2)));
        assert!(upsampled[0] == 0.0 && upsampled[3] == 1.0);
    }

    #[test]
    fn feather_keeps_inside_and_softens_edges() {
        let mut m = disc(64, 20.0);
        feather(&mut m, 64, 2.0);
        assert!(m[32 * 64 + 32] > 0.999);
        assert_eq!(m[0], 0.0);
        // The soft edge sits inside the original region, never outside it.
        assert!(m.iter().zip(disc(64, 20.0)).all(|(f, d)| *f <= d + 1e-6));
        assert!(m.iter().any(|v| *v > 0.1 && *v < 0.9));
    }

    #[test]
    fn keep_out_covers_the_whole_region() {
        let mut m = disc(64, 10.0);
        keep_out(&mut m, 64, 2.0);
        // Up to its (curved, pixelated) boundary the region is fully kept.
        assert!(m.iter().zip(disc(64, 9.0)).all(|(k, r)| r == 0.0 || *k == 0.0));
        assert_eq!(m[0], 1.0);
    }
}
