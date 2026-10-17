//! Camera pixel formats into the pipeline's interleaved BGR, in one pass and into caller-owned
//! buffers, plus a letterboxing resize for cameras that won't deliver the requested size.

use image::ImageDecoder;
use image::codecs::jpeg::JpegDecoder;
use std::io::Cursor;
use wessh_core::Frame;

/// What a camera hands over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Mjpeg,
    /// Packed 4:2:2: `Y0 U Y1 V` per pixel pair.
    Yuyv,
    /// Planar luma followed by interleaved `U V` at quarter resolution.
    Nv12,
    Rgb,
    Bgr,
    Gray,
}

/// Why a camera frame could not be converted.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("{format:?} frame has {got} bytes, {width}x{height} needs {expected}")]
    Truncated { format: PixelFormat, width: usize, height: usize, expected: usize, got: usize },
    #[error("MJPEG frame is {got:?}, the camera said {expected:?}")]
    JpegSize { expected: (usize, usize), got: (usize, usize) },
    #[error("MJPEG frame could not be decoded: {0}")]
    Jpeg(#[from] image::ImageError),
}

/// Studio-swing BT.601 YUV → RGB in 8.8 fixed point (the matrix webcams use).
mod bt601 {
    pub const LUMA_OFFSET: i32 = 16;
    pub const CHROMA_OFFSET: i32 = 128;
    pub const Y: i32 = 298;
    pub const V_TO_R: i32 = 409;
    pub const U_TO_G: i32 = 100;
    pub const V_TO_G: i32 = 208;
    pub const U_TO_B: i32 = 516;
    pub const ROUND: i32 = 128;
    pub const SHIFT: u32 = 8;
}

/// One pixel from luma and centred chroma, as BGR.
fn yuv_to_bgr(y: u8, u: i32, v: i32) -> [u8; 3] {
    use bt601::{LUMA_OFFSET, ROUND, SHIFT, U_TO_B, U_TO_G, V_TO_G, V_TO_R, Y};
    let c = Y * (i32::from(y) - LUMA_OFFSET) + ROUND;
    let channel = |value: i32| (value >> SHIFT).clamp(0, 255) as u8;
    [channel(c + U_TO_B * u), channel(c - U_TO_G * u - V_TO_G * v), channel(c + V_TO_R * v)]
}

/// Convert a `width`×`height` camera frame into `out` (`width * height * 3` BGR bytes).
pub fn to_bgr(format: PixelFormat, src: &[u8], width: usize, height: usize, out: &mut [u8]) -> Result<(), ConvertError> {
    let pixels = width * height;
    let expected = match format {
        PixelFormat::Mjpeg => 0,
        PixelFormat::Yuyv => pixels * 2,
        PixelFormat::Nv12 => pixels + pixels / 2,
        PixelFormat::Rgb | PixelFormat::Bgr => pixels * Frame::CHANNELS,
        PixelFormat::Gray => pixels,
    };
    if src.len() < expected {
        return Err(ConvertError::Truncated { format, width, height, expected, got: src.len() });
    }
    let out = &mut out[..pixels * Frame::CHANNELS];
    match format {
        PixelFormat::Mjpeg => decode_jpeg(src, width, height, out)?,
        PixelFormat::Yuyv => {
            for (quad, pair) in src.as_chunks::<4>().0.iter().zip(out.as_chunks_mut::<6>().0) {
                let [y0, u, y1, v] = *quad;
                let (u, v) = (i32::from(u) - bt601::CHROMA_OFFSET, i32::from(v) - bt601::CHROMA_OFFSET);
                pair[..3].copy_from_slice(&yuv_to_bgr(y0, u, v));
                pair[3..].copy_from_slice(&yuv_to_bgr(y1, u, v));
            }
        }
        PixelFormat::Nv12 => {
            let (luma, chroma) = src.split_at(pixels);
            for (row, out_row) in out.chunks_exact_mut(width * Frame::CHANNELS).enumerate() {
                let chroma_row = &chroma[(row / 2) * width..];
                for (x, pixel) in out_row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let uv = (x / 2) * 2;
                    let (u, v) = (i32::from(chroma_row[uv]) - bt601::CHROMA_OFFSET, i32::from(chroma_row[uv + 1]) - bt601::CHROMA_OFFSET);
                    *pixel = yuv_to_bgr(luma[row * width + x], u, v);
                }
            }
        }
        PixelFormat::Rgb => {
            for (rgb, bgr) in src.as_chunks::<3>().0.iter().zip(out.as_chunks_mut::<3>().0) {
                *bgr = [rgb[2], rgb[1], rgb[0]];
            }
        }
        PixelFormat::Bgr => out.copy_from_slice(&src[..expected]),
        PixelFormat::Gray => {
            for (&g, bgr) in src.iter().zip(out.as_chunks_mut::<3>().0) {
                *bgr = [g; 3];
            }
        }
    }
    Ok(())
}

/// Decode an MJPEG frame straight into `out`, then reorder RGB to BGR in place.
fn decode_jpeg(src: &[u8], width: usize, height: usize, out: &mut [u8]) -> Result<(), ConvertError> {
    let decoder = JpegDecoder::new(Cursor::new(src))?;
    let (w, h) = decoder.dimensions();
    if (w as usize, h as usize) != (width, height) {
        return Err(ConvertError::JpegSize { expected: (width, height), got: (w as usize, h as usize) });
    }
    if decoder.color_type() == image::ColorType::L8 {
        let pixels = width * height;
        decoder.read_image(&mut out[..pixels])?;
        for i in (0..pixels).rev() {
            let g = out[i];
            out[i * 3..i * 3 + 3].fill(g);
        }
    } else {
        decoder.read_image(out)?;
        for pixel in out.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
    }
    Ok(())
}

/// Studio-swing BT.601 RGB → YUV in 8.8 fixed point.
#[cfg(any(target_os = "linux", windows, test))]
mod bt601_inverse {
    pub const Y: [i32; 3] = [66, 129, 25];
    pub const U: [i32; 3] = [-38, -74, 112];
    pub const V: [i32; 3] = [112, -94, -18];
}

/// Convert BGR into packed YUYV (`width` must be even), chroma averaged over each pixel pair.
#[cfg(any(target_os = "linux", test))]
pub fn bgr_to_yuyv(src: &[u8], width: usize, height: usize, out: &mut [u8]) {
    use bt601::{CHROMA_OFFSET, LUMA_OFFSET, ROUND, SHIFT};
    use bt601_inverse::{U, V, Y};
    let dot = |m: [i32; 3], [b, g, r]: [i32; 3]| (m[0] * r + m[1] * g + m[2] * b + ROUND) >> SHIFT;
    let pairs = width / 2 * height;
    for (pair, quad) in src.as_chunks::<6>().0.iter().zip(out.as_chunks_mut::<4>().0).take(pairs) {
        let first = [pair[0], pair[1], pair[2]].map(i32::from);
        let second = [pair[3], pair[4], pair[5]].map(i32::from);
        let mean = std::array::from_fn(|c| (first[c] + second[c] + 1) / 2);
        let byte = |value: i32| value.clamp(0, 255) as u8;
        *quad = [
            byte(dot(Y, first) + LUMA_OFFSET),
            byte(dot(U, mean) + CHROMA_OFFSET),
            byte(dot(Y, second) + LUMA_OFFSET),
            byte(dot(V, mean) + CHROMA_OFFSET),
        ];
    }
}

/// Convert BGR into NV12 (`width` and `height` even): full-resolution luma, then `U V` pairs
/// averaged over each 2×2 block. `out` holds `width * height * 3 / 2` bytes.
#[cfg(any(windows, test))]
pub fn bgr_to_nv12(src: &[u8], width: usize, height: usize, out: &mut [u8]) {
    use bt601::{CHROMA_OFFSET, LUMA_OFFSET, ROUND, SHIFT};
    use bt601_inverse::{U, V, Y};
    let dot = |m: [i32; 3], [b, g, r]: [i32; 3]| (m[0] * r + m[1] * g + m[2] * b + ROUND) >> SHIFT;
    let byte = |value: i32| value.clamp(0, 255) as u8;
    let luma_of = |pixel: [i32; 3]| byte(dot(Y, pixel) + LUMA_OFFSET);
    let stride = width * Frame::CHANNELS;
    let (luma, chroma) = out.split_at_mut((width * height).min(out.len()));
    let row_pairs = src.chunks_exact(stride * 2).zip(luma.chunks_exact_mut(width * 2)).zip(chroma.chunks_exact_mut(width));
    for ((rows, luma_rows), chroma_row) in row_pairs {
        let (top, bottom) = rows.split_at(stride);
        let (luma_top, luma_bottom) = luma_rows.split_at_mut(width);
        let blocks = top
            .as_chunks::<6>()
            .0
            .iter()
            .zip(bottom.as_chunks::<6>().0)
            .zip(luma_top.as_chunks_mut::<2>().0.iter_mut().zip(luma_bottom.as_chunks_mut::<2>().0));
        for (((upper, lower), (luma_upper, luma_lower)), uv) in blocks.zip(chroma_row.as_chunks_mut::<2>().0) {
            let pixel = |pair: &[u8; 6], i: usize| [pair[i * 3], pair[i * 3 + 1], pair[i * 3 + 2]].map(i32::from);
            let block = [pixel(upper, 0), pixel(upper, 1), pixel(lower, 0), pixel(lower, 1)];
            *luma_upper = [luma_of(block[0]), luma_of(block[1])];
            *luma_lower = [luma_of(block[2]), luma_of(block[3])];
            let mean = std::array::from_fn(|c| (block.iter().map(|p| p[c]).sum::<i32>() + 2) / 4);
            *uv = [byte(dot(U, mean) + CHROMA_OFFSET), byte(dot(V, mean) + CHROMA_OFFSET)];
        }
    }
}

/// Fit a `src_w`×`src_h` BGR image into `dst` (`dst_w`×`dst_h`), keeping its aspect ratio and
/// filling the margins with black (bilinear, pixel-centre aligned).
pub fn letterbox(src: &[u8], src_w: usize, src_h: usize, dst: &mut [u8], dst_w: usize, dst_h: usize) {
    dst.fill(0);
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let (fit_w, fit_h) = (((src_w as f32 * scale).round() as usize).min(dst_w), ((src_h as f32 * scale).round() as usize).min(dst_h));
    let (left, top) = ((dst_w - fit_w) / 2, (dst_h - fit_h) / 2);
    let sample = |coordinate: usize, limit: usize| {
        let s = ((coordinate as f32 + 0.5) / scale - 0.5).clamp(0.0, (limit - 1) as f32);
        let base = s.floor() as usize;
        (base, (base + 1).min(limit - 1), s - base as f32)
    };
    for y in 0..fit_h {
        let (y0, y1, fy) = sample(y, src_h);
        let row = &mut dst[((top + y) * dst_w + left) * Frame::CHANNELS..][..fit_w * Frame::CHANNELS];
        for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (x0, x1, fx) = sample(x, src_w);
            let at = |px: usize, py: usize, c: usize| f32::from(src[(py * src_w + px) * Frame::CHANNELS + c]);
            *pixel = std::array::from_fn(|c| {
                let top_row = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * fx;
                let bottom_row = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * fx;
                (top_row + (bottom_row - top_row) * fy + 0.5) as u8
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_greys_and_colours() {
        // Black, white, and a pure-red pixel pair (BT.601 studio swing).
        let src = [16, 128, 235, 128, 81, 90, 81, 240];
        let mut out = [0u8; 12];
        to_bgr(PixelFormat::Yuyv, &src, 4, 1, &mut out).unwrap();
        assert_eq!(&out[..6], &[0, 0, 0, 255, 255, 255]);
        let red = &out[6..9];
        assert!(red[2] > 240 && red[0] < 15 && red[1] < 15, "{red:?}");
    }

    #[test]
    fn nv12_shares_chroma_between_four_pixels() {
        let src = [235, 235, 235, 235, 128, 128];
        let mut out = [0u8; 12];
        to_bgr(PixelFormat::Nv12, &src, 2, 2, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn rgb_is_reordered_and_short_frames_are_rejected() {
        let mut out = [0u8; 3];
        to_bgr(PixelFormat::Rgb, &[1, 2, 3], 1, 1, &mut out).unwrap();
        assert_eq!(out, [3, 2, 1]);
        assert!(matches!(to_bgr(PixelFormat::Yuyv, &[0; 3], 2, 1, &mut [0; 6]), Err(ConvertError::Truncated { .. })));
    }

    #[test]
    fn mjpeg_round_trips() {
        let mut jpeg = Vec::new();
        let colour = image::RgbImage::from_pixel(8, 8, image::Rgb([200, 100, 20]));
        colour.write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg).unwrap();
        let mut out = vec![0u8; 8 * 8 * 3];
        to_bgr(PixelFormat::Mjpeg, &jpeg, 8, 8, &mut out).unwrap();
        let [b, g, r] = [out[0], out[1], out[2]].map(i32::from);
        assert!((b - 20).abs() < 8 && (g - 100).abs() < 8 && (r - 200).abs() < 8, "{:?}", &out[..3]);
        assert!(matches!(to_bgr(PixelFormat::Mjpeg, &jpeg, 4, 4, &mut out), Err(ConvertError::JpegSize { .. })));
    }

    #[test]
    fn yuyv_round_trips() {
        let bgr = [30, 60, 200, 30, 60, 200, 255, 255, 255, 0, 0, 0];
        let mut yuyv = [0u8; 8];
        bgr_to_yuyv(&bgr, 4, 1, &mut yuyv);
        let mut back = [0u8; 12];
        to_bgr(PixelFormat::Yuyv, &yuyv, 4, 1, &mut back).unwrap();
        for (a, b) in bgr[..6].iter().zip(&back[..6]) {
            assert!((i32::from(*a) - i32::from(*b)).abs() <= 3, "{bgr:?} -> {back:?}");
        }
    }

    #[test]
    fn nv12_round_trips() {
        // A 4×2 frame: two flat 2×2 blocks, a colour and white.
        let colour = [30, 60, 200];
        let mut bgr = Vec::new();
        for _ in 0..2 {
            bgr.extend(colour.iter().chain(&colour).chain(&[255; 6]));
        }
        let mut nv12 = [0u8; 12];
        bgr_to_nv12(&bgr, 4, 2, &mut nv12);
        let mut back = [0u8; 24];
        to_bgr(PixelFormat::Nv12, &nv12, 4, 2, &mut back).unwrap();
        for (a, b) in bgr.iter().zip(&back) {
            assert!((i32::from(*a) - i32::from(*b)).abs() <= 3, "{bgr:?} -> {back:?}");
        }
    }

    #[test]
    fn letterbox_keeps_the_aspect_ratio() {
        let src = vec![200u8; 2 * 2 * 3];
        let mut dst = vec![7u8; 4 * 2 * 3];
        letterbox(&src, 2, 2, &mut dst, 4, 2);
        let column = |x: usize| dst[x * 3];
        assert_eq!((column(0), column(1), column(2), column(3)), (0, 200, 200, 0));
    }
}
