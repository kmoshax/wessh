//! CPU image kernels. Everything works on small face crops or only on the face's bounding box,
//! never on whole-frame copies, and rows are processed in parallel with rayon.

use crate::geom::Affine;
use rayon::prelude::*;

/// Interleaved 8-bit BGR image (the layout ffmpeg hands us and the models expect after a swizzle).
#[derive(Clone, Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * 3] }
    }

    pub fn from_rgb_image(img: &image::RgbImage) -> Self {
        let (w, h) = img.dimensions();
        let mut data = img.as_raw().clone();
        data.chunks_exact_mut(3).for_each(|p| p.swap(0, 2));
        Self { width: w as usize, height: h as usize, data }
    }

    pub fn to_rgb_image(&self) -> image::RgbImage {
        let mut data = self.data.clone();
        data.chunks_exact_mut(3).for_each(|p| p.swap(0, 2));
        image::RgbImage::from_raw(self.width as u32, self.height as u32, data).expect("frame size")
    }

    pub fn mirror(&mut self) {
        let w = self.width;
        self.data.par_chunks_exact_mut(w * 3).for_each(|row| {
            for x in 0..w / 2 {
                let (a, b) = (x * 3, (w - 1 - x) * 3);
                for c in 0..3 {
                    row.swap(a + c, b + c);
                }
            }
        });
    }
}

#[inline(always)]
fn sample_bilinear(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: usize, yy: usize, c: usize| src[(yy * w + xx) * 3 + c] as f32;
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let top = p(x0, y0, c) * (1.0 - fx) + p(x1, y0, c) * fx;
        let bot = p(x0, y1, c) * (1.0 - fx) + p(x1, y1, c) * fx;
        *o = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Warp `frame` into a `size`x`size` crop using `m` (frame -> crop). When the face is being
/// shrunk, several sub-samples per output pixel are averaged (cheap stand-in for INTER_AREA).
pub fn warp_crop(frame: &Frame, m: &Affine, size: usize) -> Vec<u8> {
    let inv = m.invert();
    let ss = (inv.scale().ceil() as usize).clamp(1, 4);
    let offsets: Vec<f32> = (0..ss).map(|i| (i as f32 + 0.5) / ss as f32 - 0.5).collect();
    let norm = 1.0 / (ss * ss) as f32;
    let mut out = vec![0u8; size * size * 3];
    out.par_chunks_exact_mut(size * 3).enumerate().for_each(|(v, row)| {
        for u in 0..size {
            let mut acc = [0.0f32; 3];
            for oy in &offsets {
                for ox in &offsets {
                    let [x, y] = inv.apply([u as f32 + ox, v as f32 + oy]);
                    let s = sample_bilinear(&frame.data, frame.width, frame.height, x, y);
                    acc[0] += s[0];
                    acc[1] += s[1];
                    acc[2] += s[2];
                }
            }
            for c in 0..3 {
                row[u * 3 + c] = (acc[c] * norm + 0.5) as u8;
            }
        }
    });
    out
}

/// BGR u8 crop -> NCHW f32 tensor data. `rgb` swaps channel order; value = (px/255 - mean) / std.
pub fn to_nchw(crop: &[u8], size: usize, rgb: bool, mean: f32, std: f32) -> Vec<f32> {
    let plane = size * size;
    let mut out = vec![0f32; plane * 3];
    let (a, b) = (1.0 / (255.0 * std), -mean / std);
    for (i, px) in crop.chunks_exact(3).enumerate() {
        for c in 0..3 {
            let src_c = if rgb { 2 - c } else { c };
            out[c * plane + i] = px[src_c] as f32 * a + b;
        }
    }
    out
}

/// NCHW f32 tensor data -> BGR u8 crop. value_px = clamp(v * std + mean, 0, 1) * 255.
pub fn from_nchw(t: &[f32], size: usize, rgb: bool, mean: f32, std: f32) -> Vec<u8> {
    let plane = size * size;
    let mut out = vec![0u8; plane * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        for c in 0..3 {
            let src_c = if rgb { 2 - c } else { c };
            let v = (t[src_c * plane + i] * std + mean).clamp(0.0, 1.0);
            px[c] = (v * 255.0 + 0.5) as u8;
        }
    }
    out
}

/// Feathered rectangular mask like FaceFusion's box mask (padding in percent: top, right, bottom, left).
pub fn box_mask(size: usize, blur: f32, padding: [f32; 4]) -> Vec<f32> {
    let blur_amount = (size as f32 * 0.5 * blur) as usize;
    let blur_area = (blur_amount / 2).max(1);
    let pad = |p: f32| ((size as f32 * p / 100.0) as usize).max(blur_area);
    let (top, right, bottom, left) = (pad(padding[0]), pad(padding[1]), pad(padding[2]), pad(padding[3]));
    let mut mask = vec![0f32; size * size];
    for y in top..size.saturating_sub(bottom) {
        for x in left..size.saturating_sub(right) {
            mask[y * size + x] = 1.0;
        }
    }
    if blur_amount > 0 {
        gaussian_blur(&mut mask, size, size, blur_amount as f32 * 0.25);
    }
    mask
}

pub fn gaussian_blur(buf: &mut [f32], w: usize, h: usize, sigma: f32) {
    let radius = (sigma * 3.0).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let ksum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / ksum).collect();
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, kv) in kernel.iter().enumerate() {
                let xx = (x as isize + k as isize - radius).clamp(0, w as isize - 1) as usize;
                acc += buf[y * w + xx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, kv) in kernel.iter().enumerate() {
                let yy = (y as isize + k as isize - radius).clamp(0, h as isize - 1) as usize;
                acc += tmp[yy * w + x] * kv;
            }
            buf[y * w + x] = acc;
        }
    }
}

/// Blend an aligned crop back into the frame through `mask`, touching only the face's bounding box.
/// `strength` scales the mask (1.0 = full replacement inside the mask).
pub fn paste_back(frame: &mut Frame, crop: &[u8], mask: &[f32], size: usize, m: &Affine, strength: f32) {
    let inv = m.invert();
    let s = size as f32;
    let corners = [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]].map(|p| inv.apply(p));
    let x0 = corners.iter().map(|p| p[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let y0 = corners.iter().map(|p| p[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let x1 = (corners.iter().map(|p| p[0]).fold(f32::MIN, f32::max).ceil() as usize).min(frame.width);
    let y1 = (corners.iter().map(|p| p[1]).fold(f32::MIN, f32::max).ceil() as usize).min(frame.height);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let w = frame.width;
    let max = (size - 1) as f32;
    frame.data[y0 * w * 3..y1 * w * 3].par_chunks_exact_mut(w * 3).enumerate().for_each(|(dy, row)| {
        let y = (y0 + dy) as f32;
        for x in x0..x1 {
            let [u, v] = m.apply([x as f32, y]);
            if u < 0.0 || v < 0.0 || u > max || v > max {
                continue;
            }
            let (ui, vi) = (u as usize, v as usize);
            let (u1, v1) = ((ui + 1).min(size - 1), (vi + 1).min(size - 1));
            let (fu, fv) = (u - ui as f32, v - vi as f32);
            let mk = |xx: usize, yy: usize| mask[yy * size + xx];
            let a = ((mk(ui, vi) * (1.0 - fu) + mk(u1, vi) * fu) * (1.0 - fv) + (mk(ui, v1) * (1.0 - fu) + mk(u1, v1) * fu) * fv) * strength;
            if a <= 0.002 {
                continue;
            }
            let src = sample_bilinear(crop, size, size, u, v);
            let px = &mut row[x * 3..x * 3 + 3];
            for c in 0..3 {
                px[c] = (px[c] as f32 * (1.0 - a) + src[c] * a + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Letterbox the frame into a `size`x`size` NCHW tensor for the detector (BGR, (px-127.5)/128).
/// Returns the tensor and the factor that maps detector coordinates back to frame coordinates.
pub fn detector_input(frame: &Frame, size: usize) -> (Vec<f32>, f32) {
    let scale = (size as f32 / frame.width as f32).min(size as f32 / frame.height as f32).min(1.0);
    let (rw, rh) = ((frame.width as f32 * scale) as usize, (frame.height as f32 * scale) as usize);
    let plane = size * size;
    let pad = -127.5 / 128.0;
    let mut out = vec![pad; plane * 3];
    let inv = 1.0 / scale;
    let (b, rest) = out.split_at_mut(plane);
    let (g, r) = rest.split_at_mut(plane);
    b.par_chunks_exact_mut(size)
        .zip(g.par_chunks_exact_mut(size))
        .zip(r.par_chunks_exact_mut(size))
        .take(rh)
        .enumerate()
        .for_each(|(y, ((rb, rg), rr))| {
            let sy = (y as f32 + 0.5) * inv - 0.5;
            for x in 0..rw {
                let sx = (x as f32 + 0.5) * inv - 0.5;
                let p = sample_bilinear(&frame.data, frame.width, frame.height, sx, sy);
                rb[x] = (p[0] - 127.5) / 128.0;
                rg[x] = (p[1] - 127.5) / 128.0;
                rr[x] = (p[2] - 127.5) / 128.0;
            }
        });
    (out, inv)
}

/// Fast BGR -> RGBA conversion for display.
pub fn to_rgba(frame: &Frame) -> Vec<u8> {
    let mut out = vec![255u8; frame.width * frame.height * 4];
    out.par_chunks_exact_mut(frame.width * 4).zip(frame.data.par_chunks_exact(frame.width * 3)).for_each(|(o, i)| {
        for (dst, src) in o.chunks_exact_mut(4).zip(i.chunks_exact(3)) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
        }
    });
    out
}
