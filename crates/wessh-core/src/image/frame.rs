use crate::geometry::Rect;
use rayon::prelude::*;

/// Interleaved 8-bit BGR image — the layout ffmpeg produces and OpenCV-trained models expect.
#[derive(Clone, Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frame({}x{})", self.width, self.height)
    }
}

impl Frame {
    pub const CHANNELS: usize = 3;

    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * Self::CHANNELS] }
    }

    /// Copy `rect` of this frame into `out`, reusing its buffer.
    pub fn crop_into(&self, rect: Rect, out: &mut Self) {
        let row = rect.width * Self::CHANNELS;
        out.width = rect.width;
        out.height = rect.height;
        out.data.clear();
        for y in rect.y..rect.y + rect.height {
            let start = (y * self.width + rect.x) * Self::CHANNELS;
            out.data.extend_from_slice(&self.data[start..start + row]);
        }
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.width * self.height * Self::CHANNELS
    }

    #[must_use]
    pub fn from_rgb(image: &image::RgbImage) -> Self {
        let (width, height) = image.dimensions();
        let mut data = image.as_raw().clone();
        swap_red_blue(&mut data);
        Self { width: width as usize, height: height as usize, data }
    }

    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        Ok(Self::from_rgb(&image::open(path)?.to_rgb8()))
    }

    #[must_use]
    pub fn to_rgb(&self) -> image::RgbImage {
        let mut data = self.data.clone();
        swap_red_blue(&mut data);
        image::RgbImage::from_raw(self.width as u32, self.height as u32, data).expect("buffer matches dimensions")
    }

    /// RGBA copy for display.
    #[must_use]
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_rgba(&mut out);
        out
    }

    /// RGBA copy for display into a reusable buffer (resized as needed).
    pub fn write_rgba(&self, out: &mut Vec<u8>) {
        const RGBA: usize = 4;
        let pixels = self.width * self.height;
        out.resize(pixels * RGBA, 0);
        if pixels == 0 {
            return;
        }
        // Rows are split into bands so each rayon task is large enough to amortise scheduling.
        let band = self.width * self.height.div_ceil(rayon::current_num_threads() * 4).max(1);
        out.par_chunks_mut(band * RGBA).zip(self.data.par_chunks(band * Self::CHANNELS)).for_each(|(dst, src)| {
            let (dst, _) = dst.as_chunks_mut::<RGBA>();
            let (src, _) = src.as_chunks::<{ Self::CHANNELS }>();
            for (d, &[b, g, r]) in dst.iter_mut().zip(src) {
                *d = [r, g, b, u8::MAX];
            }
        });
    }

    /// Flip horizontally in place (selfie view).
    pub fn mirror(&mut self) {
        let width = self.width;
        self.data.par_chunks_exact_mut(width * Self::CHANNELS).for_each(|row| {
            let pixels: &mut [[u8; 3]] = row.as_chunks_mut().0;
            pixels.reverse();
        });
    }
}

fn swap_red_blue(data: &mut [u8]) {
    for p in data.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::Frame;
    use crate::geometry::Rect;

    #[test]
    fn crop_into_copies_the_rect() {
        let f = Frame { width: 3, height: 2, data: (0..18).collect() };
        let mut out = Frame::default();
        f.crop_into(Rect { x: 1, y: 1, width: 2, height: 1 }, &mut out);
        assert_eq!((out.width, out.height, out.data), (2, 1, vec![12, 13, 14, 15, 16, 17]));
    }

    #[test]
    fn mirror_reverses_pixels() {
        let mut f = Frame { width: 3, height: 1, data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9] };
        f.mirror();
        assert_eq!(f.data, [7, 8, 9, 4, 5, 6, 1, 2, 3]);
    }
}
