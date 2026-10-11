use super::Frame;
use crate::geometry::Rect;

/// ITU-R BT.601 luma weights, in BGR order.
const LUMA_BGR: [f32; 3] = [0.114, 0.587, 0.299];

/// How much fine detail `rect` of `frame` holds: the mean absolute 4-neighbour Laplacian of its
/// luma (the rect's outermost pixels only serve as neighbours). Blur, downscaling and compression
/// all lower it; only compare values measured on the same picture.
#[must_use]
pub fn detail(frame: &Frame, rect: Rect) -> f32 {
    let (x0, y0) = (rect.x + 1, rect.y + 1);
    let x1 = (rect.x + rect.width).min(frame.width).saturating_sub(1);
    let y1 = (rect.y + rect.height).min(frame.height).saturating_sub(1);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let luma = |x: usize, y: usize| {
        let i = (y * frame.width + x) * Frame::CHANNELS;
        let p = &frame.data[i..i + Frame::CHANNELS];
        LUMA_BGR[0].mul_add(f32::from(p[0]), LUMA_BGR[1].mul_add(f32::from(p[1]), LUMA_BGR[2] * f32::from(p[2])))
    };
    let mut sum = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let around = luma(x - 1, y) + luma(x + 1, y) + luma(x, y - 1) + luma(x, y + 1);
            sum += 4.0f32.mul_add(luma(x, y), -around).abs();
        }
    }
    sum / ((x1 - x0) * (y1 - y0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(size: usize, cell: usize) -> Frame {
        let mut frame = Frame::new(size, size);
        for (i, pixel) in frame.data.as_chunks_mut::<{ Frame::CHANNELS }>().0.iter_mut().enumerate() {
            let on = ((i % size) / cell + (i / size) / cell).is_multiple_of(2);
            pixel.fill(if on { 200 } else { 50 });
        }
        frame
    }

    #[test]
    fn flat_regions_hold_no_detail() {
        let mut frame = Frame::new(16, 16);
        frame.data.fill(90);
        assert!(detail(&frame, Rect::full(16, 16)).abs() < 1e-6);
    }

    #[test]
    fn finer_texture_holds_more_detail() {
        let full = Rect::full(64, 64);
        assert!(detail(&checkerboard(64, 1), full) > detail(&checkerboard(64, 8), full));
    }

    #[test]
    fn degenerate_rects_are_empty() {
        assert!(detail(&checkerboard(8, 1), Rect { x: 3, y: 3, width: 2, height: 2 }).abs() < 1e-6);
        assert!(detail(&checkerboard(8, 1), Rect { x: 7, y: 7, width: 9, height: 9 }).abs() < 1e-6);
    }
}
