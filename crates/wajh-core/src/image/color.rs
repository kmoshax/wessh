//! Colour correction: Reinhard-style statistics transfer in CIELAB.
//!
//! The swap network only sees a tight crop, so its output often drifts in skin tone or
//! exposure. Matching the per-channel mean and spread of the swapped face to the original face
//! (inside the paste mask, so occluders and background don't count) removes visible seams.

use rayon::prelude::*;
use std::sync::LazyLock;

type Lab = [f32; 3];

/// Per-channel mean and standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStats {
    pub mean: Lab,
    pub std: Lab,
}

/// Mask weights below this are ignored (pixels that won't be pasted anyway).
const MIN_WEIGHT: f32 = 1.0 / 64.0;
/// Bounds on the per-channel contrast change, so a flat or noisy crop can't blow up the output.
const STD_RATIO: (f32, f32) = (0.5, 2.0);

/// Shift the colours of `crop` towards those of `reference` (same size, BGR) where `weights` is
/// set. `strength` blends between the input (`0`) and a full transfer (`1`).
pub fn match_color(crop: &mut [u8], reference: &[u8], weights: &[f32], strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let (Some(source), Some(target)) = (ColorStats::measure(crop, weights), ColorStats::measure(reference, weights)) else {
        return;
    };
    // Lab' = Lab * gain + offset: the full transfer blended with the identity.
    let mut gain = [0f32; 3];
    let mut offset = [0f32; 3];
    for c in 0..3 {
        let ratio = (target.std[c] / source.std[c].max(1e-3)).clamp(STD_RATIO.0, STD_RATIO.1);
        gain[c] = strength.mul_add(ratio - 1.0, 1.0);
        offset[c] = strength * ratio.mul_add(-source.mean[c], target.mean[c]);
    }
    crop.par_chunks_exact_mut(3).zip(weights.par_iter()).filter(|(_, w)| **w >= MIN_WEIGHT).for_each(|(px, _)| {
        let lab = bgr_to_lab([px[0], px[1], px[2]]);
        px.copy_from_slice(&lab_to_bgr(std::array::from_fn(|c| lab[c].mul_add(gain[c], offset[c]))));
    });
}

impl ColorStats {
    /// Weighted Lab statistics of a BGR crop; `None` if the weights are (almost) empty.
    #[must_use]
    pub fn measure(crop: &[u8], weights: &[f32]) -> Option<Self> {
        let zero = || ([0f64; 3], [0f64; 3], 0f64);
        let (sum, sum_sq, total) = crop
            .par_chunks_exact(3)
            .zip(weights.par_iter())
            .filter(|(_, w)| **w >= MIN_WEIGHT)
            .fold(zero, |(mut sum, mut sq, total), (px, &w)| {
                let lab = bgr_to_lab([px[0], px[1], px[2]]);
                for c in 0..3 {
                    let v = f64::from(lab[c]);
                    sum[c] += f64::from(w) * v;
                    sq[c] += f64::from(w) * v * v;
                }
                (sum, sq, total + f64::from(w))
            })
            .reduce(zero, |a, b| (std::array::from_fn(|c| a.0[c] + b.0[c]), std::array::from_fn(|c| a.1[c] + b.1[c]), a.2 + b.2));
        if total < 1.0 {
            return None;
        }
        let mean: [f64; 3] = sum.map(|s| s / total);
        Some(Self {
            mean: mean.map(|m| m as f32),
            std: std::array::from_fn(|c| (sum_sq[c] / total - mean[c] * mean[c]).max(0.0).sqrt() as f32),
        })
    }
}

// sRGB (D65) ↔ CIELAB. The gamma curves go through lookup tables; the rest is arithmetic.

const WHITE: [f32; 3] = [0.950_47, 1.0, 1.088_83];
const EPSILON: f32 = 216.0 / 24389.0;
const KAPPA: f32 = 24389.0 / 27.0;
/// Resolution of the linear → sRGB table.
const ENCODE_STEPS: usize = 4096;

static DECODE: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let v = i as f32 / 255.0;
        if v <= 0.040_45 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    })
});

static ENCODE: LazyLock<Vec<f32>> = LazyLock::new(|| {
    (0..=ENCODE_STEPS)
        .map(|i| {
            let v = i as f32 / ENCODE_STEPS as f32;
            255.0 * if v <= 0.003_130_8 { v * 12.92 } else { 1.055f32.mul_add(v.powf(1.0 / 2.4), -0.055) }
        })
        .collect()
});

fn encode(linear: f32) -> u8 {
    let x = linear.clamp(0.0, 1.0) * ENCODE_STEPS as f32;
    let i = (x as usize).min(ENCODE_STEPS - 1);
    let t = x - i as f32;
    ENCODE[i].mul_add(1.0 - t, ENCODE[i + 1] * t).round() as u8
}

fn lab_f(t: f32) -> f32 {
    if t > EPSILON { t.cbrt() } else { KAPPA.mul_add(t, 16.0) / 116.0 }
}

fn lab_f_inv(f: f32) -> f32 {
    let cube = f * f * f;
    if cube > EPSILON { cube } else { 116.0f32.mul_add(f, -16.0) / KAPPA }
}

#[must_use]
pub fn bgr_to_lab(bgr: [u8; 3]) -> Lab {
    let [b, g, r] = bgr.map(|v| DECODE[usize::from(v)]);
    let x = 0.412_456_4f32.mul_add(r, 0.357_576_1f32.mul_add(g, 0.180_437_5 * b)) / WHITE[0];
    let y = 0.212_672_9f32.mul_add(r, 0.715_152_2f32.mul_add(g, 0.072_175 * b));
    let z = 0.019_333_9f32.mul_add(r, 0.119_192f32.mul_add(g, 0.950_304_1 * b)) / WHITE[2];
    let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
    [116.0f32.mul_add(fy, -16.0), 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[must_use]
pub fn lab_to_bgr(lab: Lab) -> [u8; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let x = lab_f_inv(lab[1].mul_add(1.0 / 500.0, fy)) * WHITE[0];
    let y = lab_f_inv(fy);
    let z = lab_f_inv(lab[2].mul_add(-1.0 / 200.0, fy)) * WHITE[2];
    let r = 3.240_454_2f32.mul_add(x, (-1.537_138_5f32).mul_add(y, -0.498_531_4 * z));
    let g = (-0.969_266f32).mul_add(x, 1.876_010_8f32.mul_add(y, 0.041_556 * z));
    let b = 0.055_643_4f32.mul_add(x, (-0.204_025_9f32).mul_add(y, 1.057_225_2 * z));
    [encode(b), encode(g), encode(r)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_round_trip_is_lossless() {
        for bgr in [[0, 0, 0], [255, 255, 255], [30, 90, 200], [200, 10, 60], [1, 2, 3], [128, 128, 128]] {
            assert_eq!(lab_to_bgr(bgr_to_lab(bgr)), bgr, "{bgr:?}");
        }
        let white = bgr_to_lab([255, 255, 255]);
        assert!((white[0] - 100.0).abs() < 0.1 && white[1].abs() < 0.1 && white[2].abs() < 0.1, "{white:?}");
    }

    fn two_tone(a: [u8; 3], b: [u8; 3], n: usize) -> Vec<u8> {
        (0..n).flat_map(|i| if i % 2 == 0 { a } else { b }).collect()
    }

    #[test]
    fn full_transfer_matches_reference_statistics() {
        let mut crop = two_tone([40, 60, 90], [60, 80, 120], 64);
        let reference = two_tone([90, 120, 170], [120, 160, 200], 64);
        let weights = vec![1.0; 64];
        match_color(&mut crop, &reference, &weights, 1.0);
        let (got, want) = (ColorStats::measure(&crop, &weights).unwrap(), ColorStats::measure(&reference, &weights).unwrap());
        for c in 0..3 {
            assert!((got.mean[c] - want.mean[c]).abs() < 1.0, "{got:?} vs {want:?}");
            assert!((got.std[c] - want.std[c]).abs() < 1.0, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn zero_strength_and_masked_pixels_are_untouched() {
        let original = two_tone([40, 60, 90], [60, 80, 120], 8);
        let reference = vec![200; 24];
        let mut crop = original.clone();
        match_color(&mut crop, &reference, &[1.0; 8], 0.0);
        assert_eq!(crop, original);
        let weights = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        match_color(&mut crop, &reference, &weights, 1.0);
        assert_eq!(crop[12..], original[12..]);
        assert_ne!(crop[..12], original[..12]);
    }
}
