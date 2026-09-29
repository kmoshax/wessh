//! Many-face budget for live video: with more faces than
//! [`Settings::max_faces_per_frame`](super::Settings::max_faces_per_frame), only that many are
//! run through the swap network each frame, taking turns; the others reuse their last swapped
//! crop. The crop lives in the aligned-face frame, so pasting it through the face's *current*
//! alignment re-warps it to where the face is now — it follows the head, only the expression
//! lags by a few frames.

use crate::face::Face;
use crate::geometry::Point;

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
}

/// Decides which faces are swapped afresh and keeps every face's last swapped crop.
#[derive(Debug, Default)]
pub struct SwapReuse {
    slots: Vec<Slot>,
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
                slot.valid = false;
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

    /// Remember the swap output for face `index`.
    pub fn store(&mut self, index: usize, crop: &[u8], crop_size: usize) {
        let slot = &mut self.slots[index];
        slot.crop.clear();
        slot.crop.extend_from_slice(crop);
        (slot.crop_size, slot.valid) = (crop_size, true);
    }

    /// Copy face `index`'s last swap output into `out`.
    pub fn load(&self, index: usize, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.slots[index].crop);
    }
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
            reuse.store(i, &[i as u8; 48], 4);
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
