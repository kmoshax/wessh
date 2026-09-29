use super::Frame;
use super::sample::{Bgr, Plane, Tap};
use crate::geometry::Affine;
use rayon::prelude::*;

/// Mask values at or below this are treated as fully transparent (below one 8-bit step).
const MIN_ALPHA: f32 = 1.0 / 512.0;
/// Mask values at or above this replace the pixel outright.
const MAX_ALPHA: f32 = 1.0 - MIN_ALPHA;

/// Alpha-blend an aligned crop back into `frame` through `mask`, visiting only the face's
/// bounding box. `opacity` scales the mask (1.0 replaces the face fully inside it).
pub fn paste_back(frame: &mut Frame, crop: &[u8], mask: &[f32], size: usize, to_crop: &Affine, opacity: f32) {
    let (Some(crop), Some(mask)) = (Bgr::new(crop, size, size), Plane::new(mask, size)) else {
        return;
    };
    let Some((x0, y0, x1, y1)) = bounding_box(frame, size, to_crop) else {
        return;
    };
    let limit = (size - 1) as f32;
    let row_len = frame.width * Frame::CHANNELS;
    let [a, b, tx, c, d, ty] = to_crop.0;

    frame.data[y0 * row_len..y1 * row_len].par_chunks_exact_mut(row_len).enumerate().for_each(|(dy, row)| {
        let y = (y0 + dy) as f32;
        // Along a row the crop coordinates are linear in x: u = a·x + row_u, v = c·x + row_v.
        let (row_u, row_v) = (b * y + tx, d * y + ty);
        let Some((start, end)) = span_inside(a, row_u, limit, x0, x1).and_then(|s| span_inside(c, row_v, limit, s.0, s.1)) else {
            return;
        };
        let pixels = &mut row.as_chunks_mut::<3>().0[start..end];
        for (x, pixel) in (start..end).zip(pixels) {
            let x = x as f32;
            let (tu, tv) = (Tap::new(a * x + row_u, size), Tap::new(c * x + row_v, size));
            let alpha = mask.sample_taps(tu, tv) * opacity;
            if alpha <= MIN_ALPHA {
                continue;
            }
            let src = crop.sample_taps(tu, tv);
            if alpha >= MAX_ALPHA {
                *pixel = src.map(|s| (s + 0.5) as u8);
            } else {
                *pixel = std::array::from_fn(|i| (f32::from(pixel[i]) * (1.0 - alpha) + src[i] * alpha + 0.5) as u8);
            }
        }
    });
}

/// Sub-range of the columns `from..to` where `slope·x + offset` lies in `[0, limit]`.
///
/// The ends are rounded outwards and the sampler clamps anyway, so pixels exactly on the crop
/// border are never dropped.
fn span_inside(slope: f32, offset: f32, limit: f32, from: usize, to: usize) -> Option<(usize, usize)> {
    const FLAT: f32 = 1e-9;
    let (lo, hi) = if slope.abs() < FLAT {
        if !(0.0..=limit).contains(&offset) {
            return None;
        }
        (from as f32, to as f32)
    } else {
        let (p, q) = (-offset / slope, (limit - offset) / slope);
        (p.min(q), p.max(q))
    };
    let start = lo.ceil().max(from as f32) as usize;
    let end = (hi.floor() + 1.0).clamp(0.0, to as f32) as usize;
    (start < end).then_some((start, end))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_covers_exactly_the_valid_columns() {
        // u = 0.5·x - 2 is in [0, 3] for x in [4, 10].
        assert_eq!(span_inside(0.5, -2.0, 3.0, 0, 100), Some((4, 11)));
        // Negative slope: u = -x + 5 is in [0, 3] for x in [2, 5].
        assert_eq!(span_inside(-1.0, 5.0, 3.0, 0, 100), Some((2, 6)));
        // Clipped by the caller's range.
        assert_eq!(span_inside(0.5, -2.0, 3.0, 6, 8), Some((6, 8)));
        // Flat and outside.
        assert_eq!(span_inside(0.0, 4.0, 3.0, 0, 100), None);
        assert_eq!(span_inside(0.0, 1.0, 3.0, 0, 100), Some((0, 100)));
    }

    #[test]
    fn opaque_paste_replaces_the_face_region() {
        let mut frame = Frame::new(8, 8);
        let crop = vec![200u8; 4 * 4 * 3];
        let mask = vec![1.0f32; 4 * 4];
        // Crop occupies frame pixels 2..=5 in both axes.
        paste_back(&mut frame, &crop, &mask, 4, &Affine([1.0, 0.0, -2.0, 0.0, 1.0, -2.0]), 1.0);
        let at = |x: usize, y: usize| frame.data[(y * 8 + x) * 3];
        assert_eq!((at(2, 2), at(5, 5), at(1, 1), at(6, 6)), (200, 200, 0, 0));
    }

    #[test]
    fn half_opacity_blends() {
        let mut frame = Frame::new(4, 4);
        frame.data.fill(100);
        paste_back(&mut frame, &[200u8; 48], &[1.0; 16], 4, &Affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 0.5);
        assert_eq!(frame.data[0], 150);
    }
}
