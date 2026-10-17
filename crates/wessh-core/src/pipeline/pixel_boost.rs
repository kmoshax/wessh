//! Pixel boost (from `FaceFusion`): run a low-resolution swap network at a multiple of its native
//! size. The target is aligned at the boosted size and split into `factor²` sub-sampled tiles —
//! tile `(a, b)` holds pixels `(y·factor + a, x·factor + b)` — each a slightly shifted
//! native-size view of the face. The tiles are swapped as one batch and interleaved back, so
//! a 128 px inswapper yields a genuinely sharper 256/384/512 px face.
//!
//! [`AUTO`] boosts only faces large enough to need it: a face close to a webcam spans 400 px
//! and more, and its native 128 px swap, stretched over it, shows blocky edges (glasses, eyelids).

use crate::Result;
use crate::face::{AlignedCrop, Swapper};
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, warp_crop};

/// The swap crop size for a requested boost: a whole multiple of `native`, at least `native`.
#[must_use]
pub fn boosted_size(native: usize, requested: usize) -> usize {
    (requested / native).max(1) * native
}

/// The pixel boost that picks the size by itself (see [`choose_size`]).
pub const AUTO: usize = 0;
/// What [`AUTO`] boosts a large face to: inswapper's 128 px twice over. 4 swaps per face, 68 ms
/// instead of 16 ms on an RTX 4050; async timewarp keeps the frame rate, the expression trails
/// a little more.
pub const AUTO_SIZE: usize = 256;
/// [`AUTO`] goes back to the native size only once the face's crop has shrunk this far below
/// [`AUTO_SIZE`], so a face hovering at the threshold doesn't switch back and forth (each switch
/// costs a synchronous swap).
const AUTO_HYSTERESIS: f32 = 0.8;

/// The swap crop size for `requested` (a size, or [`AUTO`]) with the largest face's crop
/// spanning `crop_side` frame pixels (see [`crop_side`]), `current` the size used so far.
/// [`AUTO`] boosts only on a backend `fast` enough to swap four tiles per face.
#[must_use]
pub fn choose_size(native: usize, requested: usize, crop_side: f32, current: usize, fast: bool) -> usize {
    if requested != AUTO {
        return boosted_size(native, requested);
    }
    if !fast || native >= AUTO_SIZE {
        return native;
    }
    let threshold = if current >= AUTO_SIZE { AUTO_SIZE as f32 * AUTO_HYSTERESIS } else { AUTO_SIZE as f32 };
    if crop_side >= threshold { boosted_size(native, AUTO_SIZE) } else { native }
}

/// Side, in frame pixels, of the aligned swap crop of the face at `landmarks`: what a swap of
/// any size is stretched over.
#[must_use]
pub fn crop_side(landmarks: &Landmarks5) -> f32 {
    let [a, _, _, d, _, _] = Template::Arcface128.align(landmarks, 1).0;
    1.0 / a.hypot(d)
}

/// A swapped face plus the original crop it replaced (same size and alignment).
pub struct Swapped {
    pub face: AlignedCrop,
    pub original: Vec<u8>,
}

/// Swap the face at `landmarks` with a `size`² output (see [`boosted_size`]).
pub fn swap(swapper: &mut Swapper, frame: &Frame, landmarks: &Landmarks5, condition: &[f32], size: usize) -> Result<Swapped> {
    let to_crop = Template::Arcface128.align(landmarks, size);
    let original = warp_crop(frame, &to_crop, size);
    let pixels = swap_aligned(swapper, &original, size, condition)?;
    Ok(Swapped { face: AlignedCrop { pixels, size, to_crop }, original })
}

/// Swap an already-aligned `size`² BGR crop (see [`boosted_size`]).
pub fn swap_aligned(swapper: &mut Swapper, original: &[u8], size: usize, condition: &[f32]) -> Result<Vec<u8>> {
    let factor = size / swapper.kind().resolution();
    if factor <= 1 {
        return Ok(swapper.swap_crops(&[original], condition)?.swap_remove(0));
    }
    let mut tiles = swapper.swap_crops(&implode(original, size, factor), condition)?;
    equalize(&mut tiles);
    Ok(explode(&tiles, factor))
}

/// Split a `size`² BGR crop into `factor²` interleaved tiles of `(size / factor)`².
#[must_use]
pub fn implode(crop: &[u8], size: usize, factor: usize) -> Vec<Vec<u8>> {
    let tile = size / factor;
    let mut tiles = vec![Vec::with_capacity(tile * tile * 3); factor * factor];
    for y in 0..tile * factor {
        let a = y % factor;
        for x in 0..tile * factor {
            let i = (y * size + x) * 3;
            tiles[a * factor + x % factor].extend_from_slice(&crop[i..i + 3]);
        }
    }
    tiles
}

/// Give every tile the same per-channel mean, so small brightness differences between the
/// independently swapped tiles don't show up as a fine grid.
pub fn equalize(tiles: &mut [Vec<u8>]) {
    let means: Vec<[f32; 3]> = tiles
        .iter()
        .map(|t| {
            let mut sum = [0u64; 3];
            for p in t.as_chunks::<3>().0 {
                (0..3).for_each(|c| sum[c] += u64::from(p[c]));
            }
            sum.map(|s| s as f32 / (t.len() / 3) as f32)
        })
        .collect();
    let target: [f32; 3] = std::array::from_fn(|c| means.iter().map(|m| m[c]).sum::<f32>() / means.len() as f32);
    for (tile, mean) in tiles.iter_mut().zip(&means) {
        let shift: [f32; 3] = std::array::from_fn(|c| target[c] - mean[c]);
        for p in tile.as_chunks_mut::<3>().0 {
            (0..3).for_each(|c| p[c] = (f32::from(p[c]) + shift[c]).round().clamp(0.0, 255.0) as u8);
        }
    }
}

/// Inverse of [`implode`].
#[must_use]
pub fn explode(tiles: &[Vec<u8>], factor: usize) -> Vec<u8> {
    let tile = ((tiles[0].len() / 3) as f64).sqrt() as usize;
    let size = tile * factor;
    let mut crop = vec![0u8; size * size * 3];
    for (t, pixels) in tiles.iter().enumerate() {
        let (a, b) = (t / factor, t % factor);
        for (i, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
            let (y, x) = ((i / tile) * factor + a, (i % tile) * factor + b);
            crop[(y * size + x) * 3..][..3].copy_from_slice(px);
        }
    }
    crop
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_snap_to_multiples_of_the_native_resolution() {
        assert_eq!(boosted_size(128, 0), 128);
        assert_eq!(boosted_size(128, 256), 256);
        assert_eq!(boosted_size(128, 400), 384);
        assert_eq!(boosted_size(256, 256), 256);
        assert_eq!(boosted_size(256, 512), 512);
    }

    #[test]
    fn auto_boosts_large_faces_only() {
        let auto = |side, current, fast| choose_size(128, AUTO, side, current, fast);
        assert_eq!(auto(180.0, 128, true), 128);
        assert_eq!(auto(300.0, 128, true), 256);
        // Not on a slow backend, nor for models that are already 256 px.
        assert_eq!(auto(300.0, 128, false), 128);
        assert_eq!(choose_size(256, AUTO, 600.0, 256, true), 256);
        // Between the two thresholds, the size stays as it was.
        assert_eq!(auto(230.0, 256, true), 256);
        assert_eq!(auto(230.0, 128, true), 128);
        assert_eq!(auto(190.0, 256, true), 128);
        // A size asked for is kept.
        assert_eq!(choose_size(128, 384, 50.0, 128, false), 384);
    }

    #[test]
    fn the_crop_side_is_measured_in_frame_pixels() {
        let points = Template::Arcface128.points();
        // A face whose landmarks sit where a 300 px crop would put them.
        let landmarks = points.map(|[x, y]| [x * 300.0 + 40.0, y * 300.0 + 10.0]);
        assert!((crop_side(&landmarks) - 300.0).abs() < 0.5);
    }

    #[test]
    fn implode_subsamples_and_explode_restores() {
        let size = 6;
        let crop: Vec<u8> = (0..size * size * 3).map(|i| (i * 7 % 251) as u8).collect();
        for factor in [1, 2, 3] {
            let tiles = implode(&crop, size, factor);
            assert_eq!(tiles.len(), factor * factor);
            assert_eq!(explode(&tiles, factor), crop);
        }
        // Tile (1, 0) of a 2× split starts at pixel (y = 1, x = 0).
        let tiles = implode(&crop, size, 2);
        assert_eq!(tiles[2][..3], crop[size * 3..size * 3 + 3]);
        assert_eq!(tiles[1][..3], crop[3..6]);
    }

    #[test]
    fn equalize_aligns_tile_means() {
        let mut tiles = vec![[100u8, 50, 10].repeat(4), [110u8, 60, 20].repeat(4)];
        equalize(&mut tiles);
        assert_eq!(tiles[0], tiles[1]);
        assert_eq!(tiles[0][..3], [105, 55, 15]);
    }
}
