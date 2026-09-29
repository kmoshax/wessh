//! Conversions between BGR crops and NCHW `f32` tensors.

use super::{Frame, sample};
use rayon::prelude::*;

/// `tensor = (pixel / 255 - mean) / std`, optionally swapping BGR → RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    pub mean: f32,
    pub std: f32,
    pub rgb: bool,
}

impl Normalization {
    /// `[0, 1]`, RGB.
    pub const UNIT_RGB: Self = Self { mean: 0.0, std: 1.0, rgb: true };
    /// `[-1, 1]`, RGB.
    pub const SIGNED_RGB: Self = Self { mean: 0.5, std: 0.5, rgb: true };

    const fn channel(self, c: usize) -> usize {
        if self.rgb { 2 - c } else { c }
    }
}

/// BGR crop → NCHW tensor data.
#[must_use]
pub fn to_nchw(crop: &[u8], size: usize, norm: Normalization) -> Vec<f32> {
    let plane = size * size;
    let (scale, bias) = (1.0 / (255.0 * norm.std), -norm.mean / norm.std);
    let mut out = vec![0f32; plane * 3];
    for (i, px) in crop.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = f32::from(px[norm.channel(c)]).mul_add(scale, bias);
        }
    }
    out
}

/// NCHW tensor data → BGR crop (inverse of [`to_nchw`], clamped to the valid range).
#[must_use]
pub fn from_nchw(tensor: &[f32], size: usize, norm: Normalization) -> Vec<u8> {
    let plane = size * size;
    let mut out = vec![0u8; plane * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        for (c, value) in px.iter_mut().enumerate() {
            let v = tensor[norm.channel(c) * plane + i].mul_add(norm.std, norm.mean).clamp(0.0, 1.0);
            *value = (v * 255.0 + 0.5) as u8;
        }
    }
    out
}

/// Letterbox `frame` into a `size`×`size` BGR tensor normalised as `(px - 127.5) / 128`
/// (SCRFD's convention), padding right/bottom. Returns the tensor and the factor mapping
/// tensor coordinates back to frame coordinates.
#[must_use]
pub fn letterbox_nchw(frame: &Frame, size: usize) -> (Vec<f32>, f32) {
    const MEAN: f32 = 127.5;
    const STD: f32 = 128.0;
    let scale = (size as f32 / frame.width as f32).min(size as f32 / frame.height as f32).min(1.0);
    let (width, height) = ((frame.width as f32 * scale) as usize, (frame.height as f32 * scale) as usize);
    let to_frame = 1.0 / scale;
    let plane = size * size;
    let mut out = vec![-MEAN / STD; plane * 3];

    let (b, rest) = out.split_at_mut(plane);
    let (g, r) = rest.split_at_mut(plane);
    b.par_chunks_exact_mut(size)
        .zip(g.par_chunks_exact_mut(size))
        .zip(r.par_chunks_exact_mut(size))
        .take(height)
        .enumerate()
        .for_each(|(y, ((b, g), r))| {
            let sy = (y as f32 + 0.5).mul_add(to_frame, -0.5);
            for x in 0..width {
                let sx = (x as f32 + 0.5).mul_add(to_frame, -0.5);
                let [pb, pg, pr] = sample::bilinear(&frame.data, frame.width, frame.height, sx, sy);
                b[x] = (pb - MEAN) / STD;
                g[x] = (pg - MEAN) / STD;
                r[x] = (pr - MEAN) / STD;
            }
        });
    (out, to_frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchw_round_trip() {
        let crop: Vec<u8> = (0..2 * 2 * 3).map(|i| (i * 20) as u8).collect();
        for norm in [Normalization::UNIT_RGB, Normalization::SIGNED_RGB] {
            assert_eq!(from_nchw(&to_nchw(&crop, 2, norm), 2, norm), crop);
        }
    }
}
