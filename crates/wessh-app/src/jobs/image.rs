//! Swap faces in one photo and prepare display copies for the comparison view.

use crate::live::Image;
use anyhow::{Context, Result};
use image::{RgbImage, imageops};
use std::path::Path;
use wessh_core::{Frame, Processor, Settings, Stats};

/// Longest side of the textures shown on screen; saving always uses full resolution.
pub const DISPLAY_MAX: u32 = 2048;
/// Longest side of progress thumbnails sent while a video renders.
pub const THUMBNAIL_MAX: usize = 640;
const RGBA_CHANNELS: usize = 4;

/// A swapped photo: full-resolution result plus before/after display copies.
#[derive(Clone, Debug)]
pub struct SwappedImage {
    pub result: RgbImage,
    pub before: Image,
    pub after: Image,
    pub stats: Stats,
}

/// Swap faces in the photo at `target`.
pub fn swap(processor: &mut Processor, settings: &Settings, target: &Path) -> Result<SwappedImage> {
    let original = image::open(target).with_context(|| format!("could not open {}", target.display()))?.to_rgb8();
    let mut frame = Frame::from_rgb(&original);
    let stats = processor.process(&mut frame, settings, false)?;
    let result = frame.to_rgb();
    Ok(SwappedImage { before: display(&original), after: display(&result), result, stats })
}

/// An RGBA copy no larger than [`DISPLAY_MAX`] on its longest side.
#[must_use]
pub fn display(rgb: &RgbImage) -> Image {
    let (width, height) = fit(rgb.width(), rgb.height(), DISPLAY_MAX);
    let rgba = if (width, height) == rgb.dimensions() {
        image::DynamicImage::ImageRgb8(rgb.clone()).into_rgba8()
    } else {
        image::DynamicImage::ImageRgb8(imageops::resize(rgb, width, height, imageops::FilterType::Triangle)).into_rgba8()
    };
    Image { width: width as usize, height: height as usize, rgba: rgba.into_raw() }
}

/// Size of `width × height` scaled down (never up) so its longest side is at most `max`.
#[must_use]
pub fn fit(width: u32, height: u32, max: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max {
        return (width, height);
    }
    let scale = |side: u32| ((u64::from(side) * u64::from(max)).div_ceil(u64::from(longest)) as u32).max(1);
    (scale(width), scale(height))
}

/// A nearest-neighbour RGBA thumbnail of a BGR frame (cheap enough to take a few times a second).
#[must_use]
pub fn thumbnail(frame: &Frame, max: usize) -> Image {
    let step = frame.width.max(frame.height).div_ceil(max).max(1);
    let (width, height) = (frame.width.div_ceil(step), frame.height.div_ceil(step));
    let mut rgba = Vec::with_capacity(width * height * RGBA_CHANNELS);
    for y in (0..frame.height).step_by(step) {
        let row = &frame.data[y * frame.width * Frame::CHANNELS..(y + 1) * frame.width * Frame::CHANNELS];
        for &[b, g, r] in row.as_chunks::<{ Frame::CHANNELS }>().0.iter().step_by(step) {
            rgba.extend_from_slice(&[r, g, b, u8::MAX]);
        }
    }
    Image { width, height, rgba }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_only_shrinks() {
        assert_eq!(fit(640, 480, 2048), (640, 480));
        assert_eq!(fit(4096, 2048, 2048), (2048, 1024));
        assert_eq!(fit(1000, 3000, 300), (100, 300));
    }

    #[test]
    fn thumbnail_samples_and_converts_to_rgba() {
        let mut frame = Frame::new(4, 2);
        frame.data[..3].copy_from_slice(&[1, 2, 3]); // BGR of the top-left pixel
        let thumb = thumbnail(&frame, 2);
        assert_eq!((thumb.width, thumb.height), (2, 1));
        assert_eq!(&thumb.rgba[..4], &[3, 2, 1, 255]);
        assert_eq!(thumb.rgba.len(), 2 * RGBA_CHANNELS);
    }
}
