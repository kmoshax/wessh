use super::{Frame, sample};
use crate::geometry::Affine;
use rayon::prelude::*;

/// Alpha-blend an aligned crop back into `frame` through `mask`, visiting only the face's
/// bounding box. `opacity` scales the mask (1.0 replaces the face fully inside it).
pub fn paste_back(frame: &mut Frame, crop: &[u8], mask: &[f32], size: usize, to_crop: &Affine, opacity: f32) {
    let Some((x0, y0, x1, y1)) = bounding_box(frame, size, to_crop) else {
        return;
    };
    let limit = (size - 1) as f32;
    let row_len = frame.width * Frame::CHANNELS;

    frame.data[y0 * row_len..y1 * row_len].par_chunks_exact_mut(row_len).enumerate().for_each(|(dy, row)| {
        let y = (y0 + dy) as f32;
        for x in x0..x1 {
            let [u, v] = to_crop.apply([x as f32, y]);
            if !(0.0..=limit).contains(&u) || !(0.0..=limit).contains(&v) {
                continue;
            }
            let alpha = sample::bilinear_plane(mask, size, u, v) * opacity;
            if alpha <= 1.0 / 512.0 {
                continue;
            }
            let src = sample::bilinear(crop, size, size, u, v);
            for (dst, s) in row[x * 3..x * 3 + 3].iter_mut().zip(src) {
                *dst = f32::from(*dst).mul_add(1.0 - alpha, s * alpha).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Frame-space bounding box of the crop, clipped to the frame.
fn bounding_box(frame: &Frame, size: usize, to_crop: &Affine) -> Option<(usize, usize, usize, usize)> {
    let to_frame = to_crop.invert();
    let s = size as f32;
    let corners = [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]].map(|p| to_frame.apply(p));
    let (min_x, max_x) = corners.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    let (min_y, max_y) = corners.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
    let x0 = min_x.floor().max(0.0) as usize;
    let y0 = min_y.floor().max(0.0) as usize;
    let x1 = (max_x.ceil().max(0.0) as usize).min(frame.width);
    let y1 = (max_y.ceil().max(0.0) as usize).min(frame.height);
    (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
}
