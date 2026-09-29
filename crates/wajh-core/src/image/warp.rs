use super::{Frame, sample};
use crate::geometry::Affine;
use rayon::prelude::*;

/// Warp `frame` into a `size`×`size` BGR crop through `to_crop` (frame → crop).
///
/// When the face is shrunk, each output pixel averages a grid of sub-samples — a cheap stand-in
/// for OpenCV's `INTER_AREA` that avoids aliasing on large faces.
#[must_use]
pub fn warp_crop(frame: &Frame, to_crop: &Affine, size: usize) -> Vec<u8> {
    let to_frame = to_crop.invert();
    let grid = (to_frame.scale().ceil() as usize).clamp(1, 4);
    let offsets: Vec<f32> = (0..grid).map(|i| (i as f32 + 0.5) / grid as f32 - 0.5).collect();
    let weight = 1.0 / (grid * grid) as f32;

    let mut out = vec![0u8; size * size * 3];
    out.par_chunks_exact_mut(size * 3).enumerate().for_each(|(v, row)| {
        for (u, pixel) in row.chunks_exact_mut(3).enumerate() {
            let mut acc = [0.0f32; 3];
            for dy in &offsets {
                for dx in &offsets {
                    let [x, y] = to_frame.apply([u as f32 + dx, v as f32 + dy]);
                    let s = sample::bilinear(&frame.data, frame.width, frame.height, x, y);
                    acc.iter_mut().zip(s).for_each(|(a, s)| *a += s);
                }
            }
            for (p, a) in pixel.iter_mut().zip(acc) {
                *p = (a * weight + 0.5) as u8;
            }
        }
    });
    out
}
