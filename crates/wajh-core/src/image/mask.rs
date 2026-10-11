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

/// Radius of [`face_mask`]'s ellipse, as a fraction of the crop (Deep-Live-Cam's `0.44`).
const ELLIPSE_RADIUS: f32 = 0.44;
/// Gaussian sigma per unit of blur, as a fraction of the crop. Deep-Live-Cam blurs its 128 px
/// ellipse with `GaussianBlur((31, 31), 12)`: the 31-tap window truncates the kernel to an
/// effective sigma of 8 px, which keeps the face opaque much further out than a full sigma 12.
/// The default blur of `0.3` reproduces that (0.3 × 0.2083 × 128 = 8).
const ELLIPSE_SIGMA_PER_BLUR: f32 = 0.2083;

/// Soft elliptical face mask for pasting a swapped face: opaque in the middle and fading out well
/// inside the crop, so no straight crop edge ever shows on the cheeks, jaw or forehead the way a
/// rectangular mask's does. `padding` additionally cuts in from each side.
#[must_use]
pub fn face_mask(size: usize, blur: f32, padding: Padding) -> Vec<f32> {
    let center = (size as f32 - 1.0) / 2.0;
    let radius = size as f32 * ELLIPSE_RADIUS;
    let mut mask: Vec<f32> = (0..size * size)
        .map(|i| {
            let (x, y) = ((i % size) as f32 - center, (i / size) as f32 - center);
            if x.hypot(y) <= radius { 1.0 } else { 0.0 }
        })
        .collect();
    let sigma = size as f32 * blur * ELLIPSE_SIGMA_PER_BLUR;
    if sigma > 0.0 {
        gaussian_blur(&mut mask, size, size, sigma);
    }
    if padding.iter().any(|&p| p > 0.0) {
        for (m, b) in mask.iter_mut().zip(box_mask(size, 0.0, padding)) {
            *m = m.min(b);
        }
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
mod face_mask_tests {
    use super::face_mask;

    #[test]
    fn ellipse_is_opaque_in_the_middle_and_clear_in_the_corners() {
        let size = 128;
        let m = face_mask(size, 0.3, [0.0; 4]);
        assert!(m[64 * size + 64] > 0.99);
        assert!(m[0] < 0.02 && m[size * size - 1] < 0.02, "no crop corner may show");
        // A wide, soft fade: the face is already mostly transparent at the crop's side edge.
        assert!(m[64 * size] < 0.5 && m[64 * size + 32] > 0.9);
    }

    #[test]
    fn padding_cuts_the_ellipse() {
        let size = 128;
        let m = face_mask(size, 0.3, [0.0, 0.0, 40.0, 0.0]);
        assert!(m[100 * size + 64] < 0.01, "bottom 40% removed");
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
