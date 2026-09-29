use super::sample::Bgr;
use super::Frame;
use crate::geometry::Affine;
use rayon::prelude::*;

/// Most sub-samples per axis used to anti-alias a shrinking warp.
const MAX_GRID: usize = 4;

/// Warp `frame` into a `size`×`size` BGR crop through `to_crop` (frame → crop).
///
/// When the face is shrunk, each output pixel averages a grid of sub-samples — a cheap stand-in
/// for OpenCV's `INTER_AREA` that avoids aliasing on large faces.
#[must_use]
pub fn warp_crop(frame: &Frame, to_crop: &Affine, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size * size * Frame::CHANNELS];
    warp_crop_into(frame, to_crop, size, &mut out);
    out
}

/// [`warp_crop`] into a caller-owned buffer of `size * size * 3` bytes.
pub fn warp_crop_into(frame: &Frame, to_crop: &Affine, size: usize, out: &mut [u8]) {
    let row_len = size * Frame::CHANNELS;
    assert_eq!(out.len(), size * row_len, "output buffer must hold a size×size BGR crop");
    let Some(image) = Bgr::new(&frame.data, frame.width, frame.height) else {
        out.fill(0);
        return;
    };
    let to_frame = to_crop.invert();
    let [a, b, tx, c, d, ty] = to_frame.0;
    let grid = (to_frame.scale().ceil() as usize).clamp(1, MAX_GRID);
    let weight = 1.0 / (grid * grid) as f32;
    // Frame-space offset of every sub-sample relative to the pixel's own position.
    let taps: Vec<[f32; 2]> = (0..grid * grid)
        .map(|i| {
            let offset = |k: usize| (k as f32 + 0.5) / grid as f32 - 0.5;
            let (du, dv) = (offset(i % grid), offset(i / grid));
            [a * du + b * dv, c * du + d * dv]
        })
        .collect();

    out.par_chunks_exact_mut(row_len).enumerate().for_each(|(v, row)| {
        let v = v as f32;
        let (row_x, row_y) = (b * v + tx, d * v + ty);
        for (u, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let u = u as f32;
            let (x, y) = (a * u + row_x, c * u + row_y);
            let mut acc = [0.0f32; 3];
            for &[ox, oy] in &taps {
                let s = image.sample(x + ox, y + oy);
                acc[0] += s[0];
                acc[1] += s[1];
                acc[2] += s[2];
            }
            *pixel = acc.map(|a| (a * weight + 0.5) as u8);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_warp_copies_pixels() {
        let mut frame = Frame::new(4, 4);
        frame.data.iter_mut().enumerate().for_each(|(i, p)| *p = (i * 5) as u8);
        let crop = warp_crop(&frame, &Affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 4);
        assert_eq!(crop, frame.data);
    }

    #[test]
    fn shrinking_averages_a_checkerboard_to_grey() {
        let mut frame = Frame::new(64, 64);
        for (i, px) in frame.data.chunks_exact_mut(3).enumerate() {
            px.fill(if (i % 64 + i / 64) % 2 == 0 { 0 } else { 255 });
        }
        let crop = warp_crop(&frame, &Affine([0.25, 0.0, 0.0, 0.0, 0.25, 0.0]), 16);
        let middle = crop[(8 * 16 + 8) * 3];
        assert!((100..=155).contains(&middle), "{middle}");
    }
}
