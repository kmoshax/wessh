//! Resampling of square crops and masks between model resolutions.

use super::sample;
use rayon::prelude::*;

/// Resize a square single-channel plane. Integer downscales average whole blocks (no
/// aliasing); everything else is bilinear with pixel-centre alignment.
#[must_use]
pub fn resize_plane(src: &[f32], src_size: usize, dst_size: usize) -> Vec<f32> {
    let mut out = vec![0f32; dst_size * dst_size];
    resize_plane_into(src, src_size, dst_size, &mut out);
    out
}

/// [`resize_plane`] into a caller-owned `dst_size`² buffer.
pub fn resize_plane_into(src: &[f32], src_size: usize, dst_size: usize, out: &mut [f32]) {
    assert_eq!(out.len(), dst_size * dst_size, "output buffer must hold a dst_size² plane");
    if src_size == dst_size {
        out.copy_from_slice(src);
        return;
    }
    if src_size > dst_size && src_size.is_multiple_of(dst_size) {
        let k = src_size / dst_size;
        let weight = 1.0 / (k * k) as f32;
        out.par_chunks_exact_mut(dst_size).enumerate().for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                let sum: f32 = (0..k).map(|dy| src[(y * k + dy) * src_size + x * k..][..k].iter().sum::<f32>()).sum();
                *value = sum * weight;
            }
        });
        return;
    }
    let scale = src_size as f32 / dst_size as f32;
    let limit = (src_size - 1) as f32;
    out.par_chunks_exact_mut(dst_size).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5).mul_add(scale, -0.5).clamp(0.0, limit);
        for (x, value) in row.iter_mut().enumerate() {
            let sx = (x as f32 + 0.5).mul_add(scale, -0.5).clamp(0.0, limit);
            *value = sample::bilinear_plane(src, src_size, sx, sy);
        }
    });
}

/// Resize a square interleaved BGR crop (bilinear, pixel-centre aligned; downscales
/// average a grid of sub-samples like [`super::warp_crop`]).
#[must_use]
pub fn resize_crop(src: &[u8], src_size: usize, dst_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst_size * dst_size * 3];
    resize_crop_into(src, src_size, dst_size, &mut out);
    out
}

/// Most sub-samples per axis when shrinking a crop.
const MAX_GRID: usize = 4;

/// [`resize_crop`] into a caller-owned `dst_size`² BGR buffer.
pub fn resize_crop_into(src: &[u8], src_size: usize, dst_size: usize, out: &mut [u8]) {
    assert_eq!(out.len(), dst_size * dst_size * 3, "output buffer must hold a dst_size² BGR crop");
    if src_size == dst_size {
        out.copy_from_slice(src);
        return;
    }
    let scale = src_size as f32 / dst_size as f32;
    let grid = (scale.ceil() as usize).clamp(1, MAX_GRID);
    let mut offsets = [0f32; MAX_GRID];
    for (i, o) in offsets[..grid].iter_mut().enumerate() {
        *o = ((i as f32 + 0.5) / grid as f32 - 0.5) * scale;
    }
    let offsets = &offsets[..grid];
    let weight = 1.0 / (grid * grid) as f32;
    out.par_chunks_exact_mut(dst_size * 3).enumerate().for_each(|(y, row)| {
        let cy = (y as f32 + 0.5).mul_add(scale, -0.5);
        for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let cx = (x as f32 + 0.5).mul_add(scale, -0.5);
            let mut acc = [0f32; 3];
            for dy in offsets {
                for dx in offsets {
                    let s = sample::bilinear(src, src_size, src_size, cx + dx, cy + dy);
                    acc.iter_mut().zip(s).for_each(|(a, s)| *a += s);
                }
            }
            for (p, a) in pixel.iter_mut().zip(acc) {
                *p = a.mul_add(weight, 0.5) as u8;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_downscale_averages_blocks() {
        let src = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(resize_plane(&src, 4, 2), [0.75, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn plane_upscale_preserves_constants_and_ramps() {
        let up = resize_plane(&[0.25; 4], 2, 5);
        assert!(up.iter().all(|v| (v - 0.25).abs() < 1e-6));
        let ramp = resize_plane(&[0.0, 1.0, 0.0, 1.0], 2, 4);
        assert!(ramp[0] == 0.0 && ramp[3] == 1.0 && ramp[1] < ramp[2]);
    }

    #[test]
    fn crop_resize_keeps_flat_colour() {
        let src: Vec<u8> = [10u8, 20, 30].repeat(16);
        for size in [2, 7] {
            assert!(resize_crop(&src, 4, size).as_chunks::<3>().0.iter().all(|p| *p == [10, 20, 30]));
        }
    }
}
