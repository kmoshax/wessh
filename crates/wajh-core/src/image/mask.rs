/// Mask padding in percent of the crop: top, right, bottom, left.
pub type Padding = [f32; 4];

/// Feathered rectangular mask (FaceFusion's "box" mask).
#[must_use]
pub fn box_mask(size: usize, blur: f32, padding: Padding) -> Vec<f32> {
    let blur_amount = (size as f32 * 0.5 * blur) as usize;
    let edge = (blur_amount / 2).max(1);
    let inset = |percent: f32| ((size as f32 * percent / 100.0) as usize).max(edge);
    let [top, right, bottom, left] = padding.map(inset);

    let mut mask = vec![0f32; size * size];
    for y in top..size.saturating_sub(bottom) {
        mask[y * size + left..y * size + size.saturating_sub(right).max(left)].fill(1.0);
    }
    if blur_amount > 0 {
        gaussian_blur(&mut mask, size, size, blur_amount as f32 * 0.25);
    }
    mask
}

/// Separable Gaussian blur with edge clamping.
pub fn gaussian_blur(plane: &mut [f32], width: usize, height: usize, sigma: f32) {
    gaussian_blur_with(plane, width, height, sigma, &mut Vec::new(), &mut Vec::new());
}

/// [`gaussian_blur`] with caller-owned scratch buffers (the kernel and the intermediate plane),
/// which stop allocating once they have grown to size.
pub fn gaussian_blur_with(plane: &mut [f32], width: usize, height: usize, sigma: f32, kernel: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    let radius = (sigma * 3.0).ceil() as isize;
    kernel.clear();
    kernel.extend((-radius..=radius).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()));
    let sum: f32 = kernel.iter().sum();
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let clamp = |v: isize, len: usize| v.clamp(0, len as isize - 1) as usize;
    tmp.resize(width * height, 0.0);
    for y in 0..height {
        for x in 0..width {
            tmp[y * width + x] =
                kernel.iter().enumerate().map(|(k, w)| plane[y * width + clamp(x as isize + k as isize - radius, width)] * w).sum();
        }
    }
    for y in 0..height {
        for x in 0..width {
            plane[y * width + x] =
                kernel.iter().enumerate().map(|(k, w)| tmp[clamp(y as isize + k as isize - radius, height) * width + x] * w).sum();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::box_mask;

    #[test]
    fn box_mask_is_opaque_in_the_middle_and_clear_at_the_edge() {
        let m = box_mask(64, 0.3, [0.0; 4]);
        assert!(m[32 * 64 + 32] > 0.99);
        assert!(m[0] < 0.05);
    }
}
