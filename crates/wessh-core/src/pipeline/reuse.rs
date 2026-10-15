//! Every face's last swapped crop, for live video. The crop lives in the aligned-face frame, so
//! pasting it through the face's *current* alignment re-warps it to where the face is now — it
//! follows the head, only the expression lags. Two things build on that:
//!
//! - the many-face budget: with more faces than
//!   [`Settings::max_faces_per_frame`](super::Settings::max_faces_per_frame), only that many are
//!   run through the swap network each frame, taking turns;
//! - async timewarp ([`super::async_swap`]): the swap network runs beside the frame loop and
//!   every frame pastes the newest crop it has.

use crate::face::Face;
use crate::geometry::{Affine, Point};
use crate::image::{Frame, warp_crop_into};
use std::time::{Duration, Instant};

/// A face whose centre jumped by more than this many face sizes since the last frame is
/// treated as a different face (its swapped crop is dropped).
const SAME_FACE_JUMP: f32 = 0.5;

/// Last swapped crop of one face (by position in the frame's face list).
#[derive(Debug, Default)]
struct Slot {
    center: Point,
    size: f32,
    /// Swap output (before colour correction), `crop_size`² BGR.
    crop: Vec<u8>,
    crop_size: usize,
    valid: bool,
    /// Bumped whenever the slot's crop stops being usable, so a swap started before that
    /// (on another thread) is recognised as stale when it lands.
    generation: u64,
    /// When the frame the crop was swapped from was captured.
    taken: Option<Instant>,
    /// Counts the crops stored, so a consumer can tell when it changed.
    version: u64,
    /// The aligned camera crop `crop` was swapped from, and its alignment.
    source: Vec<u8>,
    to_crop: Affine,
}

/// Decides which faces are swapped afresh and keeps every face's last swapped crop.
#[derive(Debug, Default)]
pub struct SwapReuse {
    slots: Vec<Slot>,
    /// A face's region of the current frame, for [`SwapReuse::change`].
    scratch: Vec<u8>,
    /// Next face in the round-robin.
    cursor: usize,
}

impl SwapReuse {
    /// Fill `fresh` with, per face, whether to run the swap network on it this frame: all of them
    /// when `budget` is 0 or not exceeded; otherwise every face without a usable crop plus
    /// the next ones in turn, `budget` in total (more only when that many faces are new).
    pub fn plan(&mut self, faces: &[Face], budget: usize, crop_size: usize, fresh: &mut Vec<bool>) {
        fresh.clear();
        if self.slots.len() < faces.len() {
            self.slots.resize_with(faces.len(), Slot::default);
        }
        for (slot, face) in self.slots.iter_mut().zip(faces) {
            let (center, size) = (center(face), face.size());
            let jumped = (center[0] - slot.center[0]).hypot(center[1] - slot.center[1]) > SAME_FACE_JUMP * slot.size.max(size);
            if jumped || slot.crop_size != crop_size {
                slot.invalidate();
            }
            (slot.center, slot.size) = (center, size);
            fresh.push(!slot.valid);
        }
        if budget == 0 || faces.len() <= budget {
            fresh.fill(true);
            return;
        }
        let mut left = budget.saturating_sub(fresh.iter().filter(|f| **f).count());
        let mut steps = 0;
        while left > 0 && steps < faces.len() {
            let index = (self.cursor + steps) % faces.len();
            if !fresh[index] {
                fresh[index] = true;
                left -= 1;
            }
            steps += 1;
        }
        self.cursor = (self.cursor + steps) % faces.len();
    }

    /// Remember the swap output for face `index`, swapped from `source` in a frame captured just
    /// now. A swap of the face still in flight is older, so it is marked stale.
    pub fn store(&mut self, index: usize, crop: &[u8], source: &[u8], to_crop: &Affine, crop_size: usize) {
        let slot = &mut self.slots[index];
        slot.generation += 1;
        slot.fill(crop, source, to_crop, crop_size, Instant::now());
    }

    /// Remember a swap that was started at `generation` from `source` in a frame captured at
    /// `taken`; `false` (and nothing stored) when the face has changed since.
    #[expect(clippy::too_many_arguments, reason = "a landed swap's fields; one call site")]
    pub fn store_if_current(
        &mut self,
        index: usize,
        generation: u64,
        crop: &[u8],
        source: &[u8],
        to_crop: &Affine,
        crop_size: usize,
        taken: Instant,
    ) -> bool {
        match self.slots.get_mut(index) {
            Some(slot) if slot.generation == generation => {
                slot.fill(crop, source, to_crop, crop_size, taken);
                true
            }
            _ => false,
        }
    }

    /// How much face `index` changed in `frame` since the frame its swap came from (0–255; see
    /// [`largest_block_change`]), `None` without a swap. Both are cropped through the swap's own
    /// alignment, so the detector's frame-to-frame landmark jitter on a still face (a few pixels,
    /// in scale and rotation too) doesn't count — only the picture changing does.
    pub fn change(&mut self, index: usize, frame: &Frame) -> Option<f32> {
        let slot = self.slots.get(index).filter(|slot| slot.valid && !slot.source.is_empty())?;
        self.scratch.resize(slot.source.len(), 0);
        warp_crop_into(frame, &slot.to_crop, slot.crop_size, &mut self.scratch);
        Some(largest_block_change(&slot.source, &self.scratch, slot.crop_size))
    }

    /// Whether face `index` has a crop to paste.
    #[must_use]
    pub fn is_valid(&self, index: usize) -> bool {
        self.slots.get(index).is_some_and(|slot| slot.valid)
    }

    /// Face `index`'s generation, to hand to [`SwapReuse::store_if_current`].
    #[must_use]
    pub fn generation(&self, index: usize) -> u64 {
        self.slots[index].generation
    }

    /// Changes whenever face `index` gets a new crop.
    #[must_use]
    pub fn version(&self, index: usize) -> u64 {
        self.slots.get(index).map_or(0, |slot| slot.version)
    }

    /// How old face `index`'s crop is.
    #[must_use]
    pub fn age(&self, index: usize) -> Option<Duration> {
        self.slots.get(index).and_then(|slot| slot.taken).map(|taken| taken.elapsed())
    }

    /// Drop every crop: the identity or the swap model changed.
    pub fn invalidate_all(&mut self) {
        self.slots.iter_mut().for_each(Slot::invalidate);
    }

    /// The next of `faces` faces whose turn it is for a fresh swap.
    pub fn next_turn(&mut self, faces: usize) -> usize {
        let index = self.cursor % faces.max(1);
        self.cursor = index + 1;
        index
    }

    /// Copy face `index`'s last swap output into `out`.
    pub fn load(&self, index: usize, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.slots[index].crop);
    }
}

impl Slot {
    fn invalidate(&mut self) {
        self.valid = false;
        self.generation += 1;
    }

    fn fill(&mut self, crop: &[u8], source: &[u8], to_crop: &Affine, crop_size: usize, taken: Instant) {
        self.crop.clear();
        self.crop.extend_from_slice(crop);
        self.source.clear();
        self.source.extend_from_slice(source);
        (self.to_crop, self.crop_size, self.valid, self.taken) = (*to_crop, crop_size, true, Some(taken));
        self.version += 1;
    }
}

/// Side of the grey thumbnails crops are compared at (4 px cells of a 128 px crop), which also
/// averages away the camera's sensor noise.
const THUMB: usize = 32;
/// Thumbnail cells per side of a compared block (16 px of a 128 px crop: about a mouth corner),
/// so a local change isn't averaged away by the rest of the face.
const BLOCK: usize = 4;

/// How differently `a` and `b` (`size`² BGR) look: the largest mean grey difference (0–255) of
/// a block.
fn largest_block_change(a: &[u8], b: &[u8], size: usize) -> f32 {
    let (a, b) = (thumbnail(a, size), thumbnail(b, size));
    let blocks = THUMB / BLOCK;
    let mut sums = [0u32; (THUMB / BLOCK) * (THUMB / BLOCK)];
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        let (y, x) = (i / THUMB, i % THUMB);
        sums[(y / BLOCK) * blocks + x / BLOCK] += u32::from(p.abs_diff(*q));
    }
    sums.into_iter().max().unwrap_or_default() as f32 / (BLOCK * BLOCK) as f32
}

/// A [`THUMB`]² grey box-filtered copy of a `size`² BGR crop.
fn thumbnail(crop: &[u8], size: usize) -> [u8; THUMB * THUMB] {
    let mut sums = [0u32; THUMB * THUMB];
    for (y, row) in crop.chunks_exact(size * 3).take(size).enumerate() {
        let cells = &mut sums[(y * THUMB / size) * THUMB..][..THUMB];
        for (x, p) in row.as_chunks::<3>().0.iter().enumerate() {
            cells[x * THUMB / size] += u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
        }
    }
    let per_cell = (3 * size * size / (THUMB * THUMB)).max(1) as u32;
    sums.map(|sum| (sum / per_cell).min(255) as u8)
}

fn center(face: &Face) -> Point {
    [f32::midpoint(face.bbox[0], face.bbox[2]), f32::midpoint(face.bbox[1], face.bbox[3])]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_at(x: f32) -> Face {
        Face { bbox: [x - 50.0, 0.0, x + 50.0, 100.0], score: 0.9, landmarks: [[x, 50.0]; 5] }
    }

    /// Plan a frame and store a crop for every face swapped.
    fn frame(reuse: &mut SwapReuse, faces: &[Face], budget: usize) -> Vec<bool> {
        let mut fresh = Vec::new();
        reuse.plan(faces, budget, 4, &mut fresh);
        for (i, _) in fresh.iter().enumerate().filter(|(_, f)| **f) {
            reuse.store(i, &[i as u8; 48], &[0; 48], &Affine::IDENTITY, 4);
        }
        fresh
    }

    #[test]
    fn within_budget_everything_is_swapped() {
        let faces: Vec<Face> = (0..3).map(|i| face_at(200.0 * i as f32)).collect();
        let mut reuse = SwapReuse::default();
        assert_eq!(frame(&mut reuse, &faces, 0), [true; 3]);
        assert_eq!(frame(&mut reuse, &faces, 3), [true; 3]);
    }

    #[test]
    fn over_budget_faces_take_turns() {
        let faces: Vec<Face> = (0..4).map(|i| face_at(200.0 * i as f32)).collect();
        let mut reuse = SwapReuse::default();
        // The first frame has no crops to reuse.
        assert_eq!(frame(&mut reuse, &faces, 2), [true; 4]);
        let mut counts = [0; 4];
        for _ in 0..4 {
            let fresh = frame(&mut reuse, &faces, 2);
            assert_eq!(fresh.iter().filter(|f| **f).count(), 2);
            counts.iter_mut().zip(&fresh).for_each(|(c, f)| *c += usize::from(*f));
        }
        assert_eq!(counts, [2; 4]);
        let mut crop = Vec::new();
        reuse.load(3, &mut crop);
        assert_eq!(crop, [3; 48]);
    }

    #[test]
    fn swaps_that_land_after_the_face_changed_are_dropped() {
        let mut faces = vec![face_at(0.0)];
        let mut reuse = SwapReuse::default();
        frame(&mut reuse, &faces, 0);
        let (generation, taken) = (reuse.generation(0), Instant::now());
        assert!(reuse.store_if_current(0, generation, &[7; 48], &[0; 48], &Affine::IDENTITY, 4, taken));
        // The face jumps while another swap is in flight: that swap is stale.
        let stale = reuse.generation(0);
        faces[0] = face_at(1000.0);
        let mut fresh = Vec::new();
        reuse.plan(&faces, 0, 4, &mut fresh);
        assert!(!reuse.is_valid(0));
        assert!(!reuse.store_if_current(0, stale, &[9; 48], &[0; 48], &Affine::IDENTITY, 4, taken));
        reuse.invalidate_all();
        assert!(!reuse.store_if_current(0, stale + 1, &[9; 48], &[0; 48], &Affine::IDENTITY, 4, taken));
        assert!(reuse.store_if_current(0, reuse.generation(0), &[9; 48], &[0; 48], &Affine::IDENTITY, 4, taken));
        assert!(reuse.is_valid(0));
    }

    #[test]
    fn a_local_change_is_not_averaged_away() {
        let size = 128;
        let face: Vec<u8> = (0..size * size).flat_map(|i| [(if i % size < 64 { 40 } else { 200 }) + (i / size) as u8 / 4; 3]).collect();
        assert_eq!(largest_block_change(&face, &face, size), 0.0);
        // A mouth-sized patch opens: 16 px square going 60 levels darker, 1 % of the crop.
        let mut talking = face.clone();
        for y in 96..112 {
            talking[(y * size + 32) * 3..(y * size + 48) * 3].iter_mut().for_each(|v| *v = v.saturating_sub(60));
        }
        assert_eq!(largest_block_change(&face, &talking, size), 60.0);
    }

    #[test]
    fn changes_are_measured_through_the_swaps_alignment() {
        let size = 64;
        let mut frame = Frame::new(96, 96);
        frame.data.fill(90);
        let mut reuse = SwapReuse::default();
        reuse.plan(&[face_at(0.0)], 0, size, &mut Vec::new());
        assert_eq!(reuse.change(0, &frame), None);
        let to_crop = Affine([1.0, 0.0, -8.0, 0.0, 1.0, -8.0]);
        reuse.store(0, &[0; 64 * 64 * 3], &[90; 64 * 64 * 3], &to_crop, size);
        assert_eq!(reuse.change(0, &frame), Some(0.0));
        frame.data.fill(130);
        assert_eq!(reuse.change(0, &frame), Some(40.0));
    }

    #[test]
    fn turns_cycle_through_the_faces() {
        let mut reuse = SwapReuse::default();
        assert_eq!([0, 1, 2, 0].map(|_| reuse.next_turn(3)), [0, 1, 2, 0]);
        assert_eq!(reuse.next_turn(1), 0);
    }

    #[test]
    fn a_new_face_is_swapped_at_once() {
        let mut faces: Vec<Face> = (0..3).map(|i| face_at(200.0 * i as f32)).collect();
        let mut reuse = SwapReuse::default();
        frame(&mut reuse, &faces, 1);
        faces[2] = face_at(1000.0);
        let fresh = frame(&mut reuse, &faces, 1);
        assert!(fresh[2]);
        assert_eq!(fresh.iter().filter(|f| **f).count(), 1);
    }
}
