/// Bilinear sample of a 3-channel interleaved image with edge clamping (`BORDER_REPLICATE`).
#[inline]
pub(super) fn bilinear(src: &[u8], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |px: usize, py: usize, c: usize| f32::from(src[(py * width + px) * 3 + c]);
    std::array::from_fn(|c| {
        let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
        let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// Bilinear sample of a single-channel `f32` plane.
#[inline]
pub(super) fn bilinear_plane(plane: &[f32], size: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |px: usize, py: usize| plane[py * size + px];
    (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy) + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
}
