//! Conversions between BGR crops and NCHW `f32` tensors.

use super::Frame;
use crate::geometry::Rect;
use super::sample::{Bgr, Tap};
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
    let mut out = vec![0f32; size * size * 3];
    to_nchw_into(crop, size, norm, &mut out);
    out
}

/// [`to_nchw`] into a caller-owned `3 * size * size` buffer (e.g. a bound model input).
pub fn to_nchw_into(crop: &[u8], size: usize, norm: Normalization, out: &mut [f32]) {
    let plane = size * size;
    assert_eq!(out.len(), plane * 3, "output buffer must hold a 3×size×size tensor");
    let (scale, bias) = (1.0 / (255.0 * norm.std), -norm.mean / norm.std);
    let (first, rest) = out.split_at_mut(plane);
    let (second, third) = rest.split_at_mut(plane);
    let (pixels, _) = crop.as_chunks::<3>();
    for (((px, a), b), c) in pixels.iter().zip(first).zip(second).zip(third) {
        *a = f32::from(px[norm.channel(0)]) * scale + bias;
        *b = f32::from(px[norm.channel(1)]) * scale + bias;
        *c = f32::from(px[norm.channel(2)]) * scale + bias;
    }
}

/// NCHW tensor data → BGR crop (inverse of [`to_nchw`], clamped to the valid range).
#[must_use]
pub fn from_nchw(tensor: &[f32], size: usize, norm: Normalization) -> Vec<u8> {
    let plane = size * size;
    let mut out = vec![0u8; plane * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        for (c, value) in px.iter_mut().enumerate() {
            let v = (tensor[norm.channel(c) * plane + i] * norm.std + norm.mean).clamp(0.0, 1.0);
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
    let mut out = vec![0f32; size * size * 3];
    let to_frame = letterbox_nchw_into(frame, size, &mut out);
    (out, to_frame)
}

/// [`letterbox_nchw`] into a caller-owned `3 * size * size` buffer; returns the tensor → frame
/// scale factor.
pub fn letterbox_nchw_into(frame: &Frame, size: usize, out: &mut [f32]) -> f32 {
    letterbox_region_nchw_into(frame, Rect::full(frame.width, frame.height), size, out)
}

/// Letterbox only `region` of `frame` (never upscaling) into a caller-owned `3 * size * size`
/// buffer. Tensor point `p` maps to frame point `region.origin + p * factor`, where `factor` is
/// the returned value.
pub fn letterbox_region_nchw_into(frame: &Frame, region: Rect, size: usize, out: &mut [f32]) -> f32 {
    const MEAN: f32 = 127.5;
    const STD: f32 = 128.0;
    let plane = size * size;
    assert_eq!(out.len(), plane * 3, "output buffer must hold a 3×size×size tensor");
    out.fill(-MEAN / STD);
    let scale = (size as f32 / region.width as f32).min(size as f32 / region.height as f32).min(1.0);
    let to_frame = 1.0 / scale;
    let Some(image) = Bgr::new(&frame.data, frame.width, frame.height) else {
        return to_frame;
    };
    let (width, height) = ((region.width as f32 * scale) as usize, (region.height as f32 * scale) as usize);
    let (left, top) = (region.x as f32, region.y as f32);
    // The resize is axis-aligned, so every row shares the same horizontal taps.
    let column_taps: Vec<Tap> = (0..width).map(|x| Tap::new(left + (x as f32 + 0.5) * to_frame - 0.5, frame.width)).collect();

    let (b, rest) = out.split_at_mut(plane);
    let (g, r) = rest.split_at_mut(plane);
    b.par_chunks_exact_mut(size)
        .zip(g.par_chunks_exact_mut(size))
        .zip(r.par_chunks_exact_mut(size))
        .take(height)
        .enumerate()
        .for_each(|(y, ((b, g), r))| {
            let row_tap = Tap::new(top + (y as f32 + 0.5) * to_frame - 0.5, frame.height);
            for (x, &column_tap) in column_taps.iter().enumerate() {
                let [pb, pg, pr] = image.sample_taps(column_tap, row_tap);
                b[x] = (pb - MEAN) / STD;
                g[x] = (pg - MEAN) / STD;
                r[x] = (pr - MEAN) / STD;
            }
        });
    to_frame
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
