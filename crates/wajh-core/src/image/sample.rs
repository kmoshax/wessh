//! Bilinear sampling with edge clamping (OpenCV's `BORDER_REPLICATE`).
//!
//! The hot loops of every warp go through here, so the samplers are written to keep bounds
//! checks to one per image row touched: a coordinate is clamped so that its left/top neighbour
//! index is at most `len - 2`, which makes the right/bottom neighbour always valid and lets both
//! be read with a single fixed-size slice.

/// Integer neighbour index and interpolation weight along one axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Tap {
    pub index: usize,
    pub frac: f32,
}

impl Tap {
    /// Tap for coordinate `x` on an axis of `len >= 2` samples. At the far edge the weight
    /// becomes 1.0 on the last sample instead of reading past it.
    #[inline]
    pub fn new(x: f32, len: usize) -> Self {
        let x = x.clamp(0.0, (len - 1) as f32);
        let index = (x as usize).min(len - 2);
        Self { index, frac: x - index as f32 }
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}

/// An interleaved 3-channel `u8` image viewed for sampling.
#[derive(Clone, Copy, Debug)]
pub(super) struct Bgr<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> Bgr<'a> {
    const CHANNELS: usize = 3;

    /// `None` when the image is too small to interpolate (fewer than 2×2 pixels).
    pub fn new(data: &'a [u8], width: usize, height: usize) -> Option<Self> {
        (width >= 2 && height >= 2 && data.len() >= width * height * Self::CHANNELS).then_some(Self { data, width, height })
    }

    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        self.sample_taps(Tap::new(x, self.width), Tap::new(y, self.height))
    }

    #[inline]
    pub fn sample_taps(&self, tx: Tap, ty: Tap) -> [f32; 3] {
        const PAIR: usize = 2 * Bgr::CHANNELS;
        let stride = self.width * Self::CHANNELS;
        let top_at = ty.index * stride + tx.index * Self::CHANNELS;
        let bottom_at = top_at + stride;
        let top: &[u8; PAIR] = self.data[top_at..top_at + PAIR].try_into().expect("slice has PAIR bytes");
        let bottom: &[u8; PAIR] = self.data[bottom_at..bottom_at + PAIR].try_into().expect("slice has PAIR bytes");
        std::array::from_fn(|c| {
            let t = lerp(f32::from(top[c]), f32::from(top[c + Self::CHANNELS]), tx.frac);
            let b = lerp(f32::from(bottom[c]), f32::from(bottom[c + Self::CHANNELS]), tx.frac);
            lerp(t, b, ty.frac)
        })
    }
}

/// A square single-channel `f32` plane viewed for sampling.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plane<'a> {
    data: &'a [f32],
    size: usize,
}

impl<'a> Plane<'a> {
    /// `None` when the plane is smaller than 2×2.
    pub fn new(data: &'a [f32], size: usize) -> Option<Self> {
        (size >= 2 && data.len() >= size * size).then_some(Self { data, size })
    }

    #[inline]
    pub fn sample_taps(&self, tx: Tap, ty: Tap) -> f32 {
        let top_at = ty.index * self.size + tx.index;
        let bottom_at = top_at + self.size;
        let top: &[f32; 2] = self.data[top_at..top_at + 2].try_into().expect("slice has 2 samples");
        let bottom: &[f32; 2] = self.data[bottom_at..bottom_at + 2].try_into().expect("slice has 2 samples");
        lerp(lerp(top[0], top[1], tx.frac), lerp(bottom[0], bottom[1], tx.frac), ty.frac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_clamps_to_the_edges() {
        assert_eq!(Tap::new(-3.0, 4), Tap { index: 0, frac: 0.0 });
        assert_eq!(Tap::new(3.0, 4), Tap { index: 2, frac: 1.0 });
        assert_eq!(Tap::new(9.0, 4), Tap { index: 2, frac: 1.0 });
        assert_eq!(Tap::new(1.25, 4), Tap { index: 1, frac: 0.25 });
    }

    #[test]
    fn bgr_interpolates_between_pixels() {
        // 2×2: black, white / white, black
        let data = [0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        let image = Bgr::new(&data, 2, 2).expect("2x2");
        assert_eq!(image.sample(0.0, 0.0), [0.0; 3]);
        assert_eq!(image.sample(1.0, 0.0), [255.0; 3]);
        assert_eq!(image.sample(0.5, 0.5), [127.5; 3]);
        assert_eq!(image.sample(5.0, 5.0), [0.0; 3]);
        assert!(Bgr::new(&data[..3], 1, 1).is_none());
    }

    #[test]
    fn plane_interpolates() {
        let plane = Plane::new(&[0.0, 1.0, 1.0, 1.0], 2).expect("2x2");
        let v = plane.sample_taps(Tap::new(0.5, 2), Tap::new(0.5, 2));
        assert!((v - 0.75).abs() < 1e-6);
    }
}
