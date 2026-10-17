//! Landmark stabilisation for live video, by optical flow.
//!
//! The detector places each landmark to within a pixel or two of its own input; at the size a
//! tracked face is detected at, that is a few pixels of the frame, and they wander from frame to
//! frame even on a still face. A low-pass filter (One Euro, which this replaced) steadies them
//! only by making them lag behind a moving face. Here each landmark is instead *followed* from
//! the previous frame with pyramidal Lucas–Kanade flow — exact frame-to-frame motion, no lag —
//! and only nudged towards the detector, a little when they roughly agree (the detector's noise
//! averages out) and all the way when they don't (the flow lost the point, or the face was
//! detected afresh). On a 100 px face panning and zooming like a hand-held camera, One Euro
//! doubled the raw detector's landmark error (3.0 vs 1.45 px); flow cuts it to 1.02 px and keeps
//! a still face as steady (0.12 px).
//!
//! Flow runs on a small grey crop of the face, so its cost and window sizes don't depend on how
//! large the face is. The crop keeps one alignment from frame to frame — this frame's crop is the
//! next one's reference as it is — and is realigned only once the face has drifted away in it,
//! so a frame costs one crop warp and the flow itself (0.7–1 ms in all).
//!
//! The same flow holds a face the detector misses for a moment (motion blur, a hand passing,
//! a sharp turn): it is followed for up to [`HOLD_FRAMES`] frames, so it stays swapped instead
//! of flashing the real face, and dropped as soon as the flow loses it.

use crate::face::Face;
use crate::geometry::{Affine, Landmarks5, Point, Template};
use crate::image::{Frame, warp_crop_into};

/// Side of the aligned grey crop flow runs on: about 1.4 px of crop per eye-to-eye pixel at 128.
const CROP: usize = 128;
/// Pyramid levels (128, 64 and 32 px): together they follow about 12 crop pixels of motion a
/// frame — a head turned briskly at 30 fps.
const LEVELS: usize = 3;
/// Half the side of the window matched around each landmark (11 px: an eye corner's texture).
const RADIUS: isize = 5;
/// Gauss–Newton steps per level, and the step (in pixels) small enough to stop early.
const ITERATIONS: usize = 10;
const CONVERGED: f32 = 0.01;
/// Below this smallest eigenvalue of the window's structure tensor (per pixel, grey levels²)
/// the window is too flat to say where it moved.
const MIN_TEXTURE: f32 = 1.0;
/// A tracked point whose window still differs by more than this (mean grey levels) found the
/// wrong place.
const MAX_RESIDUAL: f32 = 12.0;
/// Weight of the detector's landmark against the tracked one: `MIN_PULL` while the two agree to
/// within `AGREE` (as a fraction of the face size, about the detector's jitter), rising to 1 at
/// `DISAGREE`, where the track is abandoned for the detection.
const MIN_PULL: f32 = 0.1;
const AGREE: f32 = 0.01;
const DISAGREE: f32 = 0.06;
/// The crop is realigned once a landmark lies this far (crop pixels) from where the face's own
/// alignment would put it: still well inside the flow's reach, and the face stays centred.
const MAX_DRIFT: f32 = 4.0;
/// A face whose centre moved by more than this many face sizes is a different face.
const SAME_FACE_JUMP: f32 = 0.5;
/// Frames a face the detector missed is followed by flow alone (a third of a second at 30 fps):
/// the detector's misses last a frame or two, and a face that really left shouldn't linger.
const HOLD_FRAMES: u32 = 10;
/// Landmarks of a held face that must still be followed; the rest move with them.
const MIN_HELD_LANDMARKS: usize = 3;

/// One followed face: its last landmarks and the last frame's grey crop, through `to_crop`.
#[derive(Debug, Default)]
struct Track {
    landmarks: Landmarks5,
    /// The face as last seen, for holding it.
    face: Face,
    to_crop: Affine,
    pyramid: Pyramid,
    matched: bool,
    /// Frames in a row the detector missed the face.
    missed: u32,
}

impl Track {
    /// Align the crop to `landmarks` and take it from `frame`.
    fn anchor(&mut self, frame: &Frame, landmarks: &Landmarks5, bgr: &mut Vec<u8>) {
        self.to_crop = Template::Arcface128.align(landmarks, CROP);
        grey_crop(frame, &self.to_crop, bgr, &mut self.pyramid);
    }
}

/// Follows every live face's landmarks from frame to frame.
#[derive(Debug, Default)]
pub struct LandmarkFlow {
    tracks: Vec<Track>,
    /// The current frame's crop through a track's alignment, and its pyramid.
    current: Pyramid,
    bgr: Vec<u8>,
}

impl LandmarkFlow {
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Replace each face's detected landmarks with its tracked ones, fused with the detection,
    /// and append the faces the detector missed that flow still follows (see [`HOLD_FRAMES`]).
    pub fn refine(&mut self, frame: &Frame, faces: &mut Vec<Face>) {
        self.tracks.iter_mut().for_each(|track| track.matched = false);
        for face in faces.iter_mut() {
            let size = face.size().max(1.0);
            let center = centroid(&face.landmarks);
            let nearest = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, track)| !track.matched)
                .map(|(index, track)| (index, distance(centroid(&track.landmarks), center)))
                .filter(|&(_, d)| d < SAME_FACE_JUMP * size)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(index, _)| index);
            let index = nearest.unwrap_or_else(|| {
                self.tracks.push(Track::default());
                self.tracks.len() - 1
            });
            let track = &mut self.tracks[index];
            if nearest.is_some() {
                grey_crop(frame, &track.to_crop, &mut self.bgr, &mut self.current);
                let to_frame = track.to_crop.invert();
                for (landmark, previous) in face.landmarks.iter_mut().zip(track.landmarks) {
                    let start = track.to_crop.apply(previous);
                    let Some(tracked) = follow(&track.pyramid, &self.current, start) else { continue };
                    let tracked = to_frame.apply(tracked);
                    let pull = ((distance(tracked, *landmark) / size - AGREE) / (DISAGREE - AGREE)).clamp(MIN_PULL, 1.0);
                    *landmark = [pull.mul_add(landmark[0] - tracked[0], tracked[0]), pull.mul_add(landmark[1] - tracked[1], tracked[1])];
                }
                let aligned = Template::Arcface128.align(&face.landmarks, CROP);
                let drift = face.landmarks.iter().map(|&p| distance(track.to_crop.apply(p), aligned.apply(p))).fold(0.0, f32::max);
                if drift > MAX_DRIFT {
                    track.anchor(frame, &face.landmarks, &mut self.bgr);
                } else {
                    std::mem::swap(&mut track.pyramid, &mut self.current);
                }
            } else {
                track.anchor(frame, &face.landmarks, &mut self.bgr);
            }
            track.landmarks = face.landmarks;
            track.face = face.clone();
            track.matched = true;
            track.missed = 0;
        }
        for track in self.tracks.iter_mut().filter(|track| !track.matched && track.missed < HOLD_FRAMES) {
            if let Some(face) = hold(track, frame, &mut self.bgr, &mut self.current) {
                faces.push(face);
            }
        }
        self.tracks.retain(|track| track.matched);
    }
}

/// Follow `track`'s face, which the detector missed, into `frame`: the face moved by the flow
/// of its landmarks, or `None` once too few of them can be followed.
fn hold(track: &mut Track, frame: &Frame, bgr: &mut Vec<u8>, current: &mut Pyramid) -> Option<Face> {
    grey_crop(frame, &track.to_crop, bgr, current);
    let to_frame = track.to_crop.invert();
    let moved: Vec<Option<Point>> =
        track.landmarks.iter().map(|&p| follow(&track.pyramid, current, track.to_crop.apply(p)).map(|q| to_frame.apply(q))).collect();
    let followed: Vec<Point> = track.landmarks.iter().zip(&moved).filter_map(|(p, q)| q.map(|q| [q[0] - p[0], q[1] - p[1]])).collect();
    if followed.len() < MIN_HELD_LANDMARKS {
        return None;
    }
    let shift = followed.iter().fold([0.0, 0.0], |[x, y], d| [x + d[0] / followed.len() as f32, y + d[1] / followed.len() as f32]);
    for (landmark, moved) in track.landmarks.iter_mut().zip(moved) {
        *landmark = moved.unwrap_or([landmark[0] + shift[0], landmark[1] + shift[1]]);
    }
    let [x0, y0, x1, y1] = track.face.bbox;
    track.face =
        Face { bbox: [x0 + shift[0], y0 + shift[1], x1 + shift[0], y1 + shift[1]], landmarks: track.landmarks, ..track.face.clone() };
    std::mem::swap(&mut track.pyramid, current);
    track.matched = true;
    track.missed += 1;
    Some(track.face.clone())
}

impl LandmarkFlow {
    #[cfg(test)]
    fn tracks(&self) -> usize {
        self.tracks.len()
    }
}

/// Grey images at [`LEVELS`] resolutions, finest first.
#[derive(Debug, Default)]
struct Pyramid(Vec<Vec<f32>>);

impl Pyramid {
    fn level(&self, level: usize) -> (&[f32], usize) {
        (&self.0[level], CROP >> level)
    }
}

/// `frame` through `to_crop` as a [`CROP`]² grey pyramid.
fn grey_crop(frame: &Frame, to_crop: &Affine, bgr: &mut Vec<u8>, pyramid: &mut Pyramid) {
    bgr.resize(CROP * CROP * Frame::CHANNELS, 0);
    warp_crop_into(frame, to_crop, CROP, bgr);
    pyramid.0.resize_with(LEVELS, Vec::new);
    let (finest, coarser) = pyramid.0.split_at_mut(1);
    finest[0].clear();
    finest[0].extend(
        bgr.as_chunks::<3>()
            .0
            .iter()
            .map(|p| 0.114f32.mul_add(f32::from(p[0]), 0.587f32.mul_add(f32::from(p[1]), 0.299 * f32::from(p[2])))),
    );
    let mut previous = &finest[0];
    for (level, image) in coarser.iter_mut().enumerate() {
        let side = CROP >> (level + 1);
        image.clear();
        image.extend((0..side * side).map(|i| {
            let (x, y, wide) = (2 * (i % side), 2 * (i / side), 2 * side);
            (previous[y * wide + x] + previous[y * wide + x + 1] + previous[(y + 1) * wide + x] + previous[(y + 1) * wide + x + 1]) / 4.0
        }));
        previous = image;
    }
}

/// Side of the matched window.
const WINDOW: usize = (2 * RADIUS + 1) as usize;

/// Offset from the window's centre of its pixel `i` (row-major).
fn window_offset(i: usize) -> Point {
    [(i % WINDOW) as f32 - RADIUS as f32, (i / WINDOW) as f32 - RADIUS as f32]
}

/// Where the point at `start` of `previous` went in `current` (both crops), or `None` when its
/// window is too flat to follow or it can't be found.
fn follow(previous: &Pyramid, current: &Pyramid, start: Point) -> Option<Point> {
    let mut guess = [0.0f32; 2];
    for level in (0..LEVELS).rev() {
        let scale = (1 << level) as f32;
        let (before, side) = previous.level(level);
        let (after, _) = current.level(level);
        let at = [start[0] / scale, start[1] / scale];
        // The window's values, gradients and structure tensor, from the previous image.
        let mut window = [(0.0f32, 0.0f32, 0.0f32); WINDOW * WINDOW];
        let (mut gxx, mut gxy, mut gyy) = (0.0f32, 0.0f32, 0.0f32);
        for (i, sample) in window.iter_mut().enumerate() {
            let [dx, dy] = window_offset(i);
            let (x, y) = (at[0] + dx, at[1] + dy);
            let ix = (sample_at(before, side, x + 1.0, y) - sample_at(before, side, x - 1.0, y)) / 2.0;
            let iy = (sample_at(before, side, x, y + 1.0) - sample_at(before, side, x, y - 1.0)) / 2.0;
            *sample = (sample_at(before, side, x, y), ix, iy);
            (gxx, gxy, gyy) = (ix.mul_add(ix, gxx), ix.mul_add(iy, gxy), iy.mul_add(iy, gyy));
        }
        let pixels = window.len() as f32;
        let det = gxx.mul_add(gyy, -gxy * gxy);
        let trace_half = f32::midpoint(gxx, gyy);
        let smallest = trace_half - trace_half.mul_add(trace_half, -det).max(0.0).sqrt();
        if smallest / pixels < MIN_TEXTURE {
            return None;
        }
        // How much window pixel `i` differs from where it is guessed to have moved.
        let difference = |i: usize, value: f32, motion: Point| {
            let [dx, dy] = window_offset(i);
            value - sample_at(after, side, at[0] + guess[0] + motion[0] + dx, at[1] + guess[1] + motion[1] + dy)
        };
        let mut motion = [0.0f32; 2];
        for _ in 0..ITERATIONS {
            let (mut bx, mut by) = (0.0f32, 0.0f32);
            for (i, &(value, ix, iy)) in window.iter().enumerate() {
                let d = difference(i, value, motion);
                (bx, by) = (d.mul_add(ix, bx), d.mul_add(iy, by));
            }
            let step = [gyy.mul_add(bx, -gxy * by) / det, gxx.mul_add(by, -gxy * bx) / det];
            motion = [motion[0] + step[0], motion[1] + step[1]];
            if step[0].hypot(step[1]) < CONVERGED {
                break;
            }
        }
        if level == 0 {
            let residual = window.iter().enumerate().map(|(i, &(value, _, _))| difference(i, value, motion).abs()).sum::<f32>() / pixels;
            let moved = [start[0] + guess[0] + motion[0], start[1] + guess[1] + motion[1]];
            return (residual <= MAX_RESIDUAL).then_some(moved);
        }
        guess = [2.0 * (guess[0] + motion[0]), 2.0 * (guess[1] + motion[1])];
    }
    None
}

/// Bilinear sample of a `side`² grey image, clamped to its edges.
fn sample_at(image: &[f32], side: usize, x: f32, y: f32) -> f32 {
    let max = (side - 1) as f32;
    let (x, y) = (x.clamp(0.0, max), y.clamp(0.0, max));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(side - 1), (y0 + 1).min(side - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = fx.mul_add(image[y0 * side + x1] - image[y0 * side + x0], image[y0 * side + x0]);
    let bottom = fx.mul_add(image[y1 * side + x1] - image[y1 * side + x0], image[y1 * side + x0]);
    fy.mul_add(bottom - top, top)
}

fn centroid(landmarks: &Landmarks5) -> Point {
    let (x, y) = landmarks.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
    [x / 5.0, y / 5.0]
}

fn distance(a: Point, b: Point) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured frame (smooth blobs) shifted by `dx` pixels.
    fn frame(dx: f32) -> Frame {
        let mut frame = Frame::new(400, 400);
        for y in 0..400 {
            for x in 0..400 {
                let (u, v) = (x as f32 - dx, y as f32);
                let value = 128.0 + 60.0 * (u / 9.0).sin() * (v / 7.0).cos() + 40.0 * ((u + v) / 13.0).sin();
                let i = (y * 400 + x) * 3;
                frame.data[i..i + 3].fill(value.clamp(0.0, 255.0) as u8);
            }
        }
        frame
    }

    fn face(landmarks: Landmarks5) -> Face {
        let [x, y] = centroid(&landmarks);
        Face { bbox: [x - 80.0, y - 80.0, x + 80.0, y + 80.0], score: 1.0, landmarks }
    }

    const LANDMARKS: Landmarks5 = [[160.0, 170.0], [240.0, 170.0], [200.0, 215.0], [170.0, 250.0], [230.0, 250.0]];

    #[test]
    fn follows_motion_and_damps_detector_jitter() {
        let mut flow = LandmarkFlow::default();
        flow.refine(&frame(0.0), &mut vec![face(LANDMARKS)]);
        // The picture moves 6 px right; the detector says 6 px plus 3 px of noise.
        let noisy = LANDMARKS.map(|[x, y]| [x + 9.0, y]);
        let mut faces = vec![face(noisy)];
        flow.refine(&frame(6.0), &mut faces);
        for (refined, truth) in faces[0].landmarks.iter().zip(LANDMARKS.map(|[x, y]| [x + 6.0, y])) {
            assert!(distance(*refined, truth) < 1.0, "{refined:?} vs {truth:?}");
        }
    }

    #[test]
    fn a_missed_face_is_held_while_flow_follows_it() {
        let mut flow = LandmarkFlow::default();
        let mut faces = vec![face(LANDMARKS)];
        flow.refine(&frame(0.0), &mut faces);
        // The detector misses the face, which moved 4 px.
        let mut faces = Vec::new();
        flow.refine(&frame(4.0), &mut faces);
        assert_eq!(faces.len(), 1);
        for (held, truth) in faces[0].landmarks.iter().zip(LANDMARKS.map(|[x, y]| [x + 4.0, y])) {
            assert!(distance(*held, truth) < 1.0, "{held:?} vs {truth:?}");
        }
        // Not for ever.
        for _ in 1..HOLD_FRAMES {
            faces.clear();
            flow.refine(&frame(4.0), &mut faces);
            assert_eq!(faces.len(), 1);
        }
        faces.clear();
        flow.refine(&frame(4.0), &mut faces);
        assert_eq!((faces.len(), flow.tracks()), (0, 0));
    }

    #[test]
    fn a_face_that_left_is_not_held() {
        let mut flow = LandmarkFlow::default();
        flow.refine(&frame(0.0), &mut vec![face(LANDMARKS)]);
        let mut empty = Frame::new(400, 400);
        empty.data.fill(128);
        let mut faces = Vec::new();
        flow.refine(&empty, &mut faces);
        assert_eq!(faces, []);
    }

    #[test]
    fn a_detection_far_from_the_track_wins() {
        let mut flow = LandmarkFlow::default();
        flow.refine(&frame(0.0), &mut vec![face(LANDMARKS)]);
        // The picture didn't move, but the detector puts the face 15 px away (≈ 0.09 face sizes).
        let moved = LANDMARKS.map(|[x, y]| [x + 15.0, y]);
        let mut faces = vec![face(moved)];
        flow.refine(&frame(0.0), &mut faces);
        assert_eq!(faces[0].landmarks, moved);
    }
}
