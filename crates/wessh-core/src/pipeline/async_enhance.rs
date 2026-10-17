//! Face restoration for live video, off the frame loop. GFPGAN takes far longer than a frame
//! (≈ 126 ms on an RTX 4050), so it runs on its own thread, one face at a time, and what it adds
//! is kept as a *residual* — restored minus swapped, in the restorer's 1024 px face-aligned space.
//! Every frame adds the newest residual to the face as it is now, through the face's current
//! alignment: the skin, hair and outline keep their restored detail at camera rate, and the next
//! restoration refreshes it. Once the face has changed since the restorer saw it, the eyes and
//! mouth are left out of the residual, since they will have moved, and so is every part of the
//! face that no longer looks as the restorer saw it: added there, the restored edges (glasses,
//! eyebrows, hairline) of the old pose would show beside the new ones, a ghost of the face.

use super::reuse::{THUMB, Thumbnail, thumbnail};
use crate::face::{AlignedCrop, Enhancer, Face};
use crate::geometry::{Landmarks5, Template};
use crate::image::{Frame, warp_crop_into};
use crate::{Error, Result};
use rayon::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The restorer, shared by the frame loop (still images, video) and the restoration thread.
pub(super) type SharedEnhancer = Arc<Mutex<Option<Enhancer>>>;

pub(super) fn lock(enhancer: &SharedEnhancer) -> MutexGuard<'_, Option<Enhancer>> {
    enhancer.lock().unwrap_or_else(PoisonError::into_inner)
}

const OUTPUT: usize = Enhancer::OUTPUT_SIZE;
/// Radii (as a fraction of the aligned face) of the eye and mouth regions left out of the
/// residual, and how far beyond them it fades back in.
const EYE_RADIUS: f32 = 0.075;
const MOUTH_RADII: [f32; 2] = [0.14, 0.075];
const FEATHER: f32 = 0.6;
/// A face is restored again at most this often, and only once its swap has changed. Each
/// restoration holds up the frames whose detection and swap queue behind it on the GPU, so it is
/// a trade between fresh detail and steady frames: on an RTX 4050 with a moving 1280×720 face,
/// every 200 ms gave 30 fps (95th percentile 82 ms), every 500 ms 45 fps (74 ms); synchronous
/// restoration, 7 fps.
const RESTORE_INTERVAL: Duration = Duration::from_millis(500);
/// Mean grey difference (0–255) of a 32 px cell of the aligned face between what the restorer
/// saw and the face now, below which the residual is added whole, and above which none of it.
const SAME_CELL: f32 = 4.0;
const CHANGED_CELL: f32 = 12.0;

struct Job {
    slot: usize,
    generation: u64,
    swap: u64,
    /// The face at the restorer's input size, and at its output size to subtract.
    input: Vec<u8>,
    base: Vec<u8>,
}

struct Done {
    slot: usize,
    generation: u64,
    swap: u64,
    residual: Result<Vec<i16>>,
    seen: Thumbnail,
    /// The job's buffers, handed back for the next one.
    input: Vec<u8>,
    base: Vec<u8>,
}

/// One face's last residual, and the swap it was taken from.
struct Residual {
    generation: u64,
    swap: u64,
    values: Vec<i16>,
    /// The face it was taken from.
    seen: Thumbnail,
}

/// What a face was last restored from.
#[derive(Clone, Copy)]
struct Restored {
    swap: u64,
    at: Instant,
}

pub(super) struct AsyncEnhance {
    jobs: Option<Sender<Job>>,
    done: Receiver<Done>,
    thread: Option<JoinHandle<()>>,
    busy: bool,
    residuals: Vec<Option<Residual>>,
    restored: Vec<Option<Restored>>,
    /// Next face whose turn it is.
    cursor: usize,
    /// Per-pixel weight of the residual: 0 over the eyes and mouth.
    weights: Vec<f32>,
    /// Per output column (and row), the thumbnail cells around it: see [`cell_span`].
    spans: Vec<(usize, usize, f32)>,
    spare_input: Vec<u8>,
    spare_base: Vec<u8>,
    /// The face as it is now, at the output size, and the restored face handed out.
    base: Vec<u8>,
    pixels: Vec<u8>,
}

impl AsyncEnhance {
    pub fn spawn(enhancer: SharedEnhancer) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (finished, done) = mpsc::channel();
        let thread = std::thread::Builder::new().name("restore".into()).spawn(move || {
            for job in inbox {
                let restored = match lock(&enhancer).as_mut() {
                    Some(enhancer) => enhancer.restore(&job.input),
                    None => Err(Error::UnexpectedOutput { model: "enhancer", detail: "no restorer loaded".to_owned() }),
                };
                let residual =
                    restored.map(|restored| restored.iter().zip(&job.base).map(|(&r, &b)| i16::from(r) - i16::from(b)).collect());
                let seen = thumbnail(&job.base, OUTPUT);
                let Job { slot, generation, swap, input, base } = job;
                if finished.send(Done { slot, generation, swap, residual, seen, input, base }).is_err() {
                    return;
                }
            }
        })?;
        Ok(Self {
            jobs: Some(jobs),
            done,
            thread: Some(thread),
            busy: false,
            residuals: Vec::new(),
            restored: Vec::new(),
            cursor: 0,
            weights: feature_weights(),
            spans: (0..OUTPUT).map(cell_span).collect(),
            spare_input: Vec::new(),
            spare_base: Vec::new(),
            base: Vec::new(),
            pixels: Vec::new(),
        })
    }

    /// Keep the restoration that finished, and hand the next face due to the thread. `frame` holds
    /// the swapped faces; `swaps` gives each face's identity and the version of its swap (see
    /// `SwapReuse`).
    pub fn update(&mut self, frame: &Frame, faces: &[Face], swaps: impl Fn(usize) -> (u64, u64)) -> Result<()> {
        self.residuals.resize_with(faces.len(), || None);
        self.restored.resize(faces.len(), None);
        match self.done.try_recv() {
            Ok(done) => {
                self.busy = false;
                (self.spare_input, self.spare_base) = (done.input, done.base);
                let values = done.residual?;
                if let Some(slot) = self.residuals.get_mut(done.slot) {
                    *slot = Some(Residual { generation: done.generation, swap: done.swap, values, seen: done.seen });
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.busy = false,
        }
        if self.busy || faces.is_empty() {
            return Ok(());
        }
        let slot = self.cursor % faces.len();
        self.cursor = slot + 1;
        let (generation, swap) = swaps(slot);
        let current = self.residuals[slot].as_ref().is_some_and(|r| r.generation == generation);
        if current && self.restored[slot].is_some_and(|r| r.swap == swap || r.at.elapsed() < RESTORE_INTERVAL) {
            return Ok(());
        }
        self.restored[slot] = Some(Restored { swap, at: Instant::now() });
        let landmarks = &faces[slot].landmarks;
        let mut input = std::mem::take(&mut self.spare_input);
        input.resize(Enhancer::INPUT_SIZE * Enhancer::INPUT_SIZE * Frame::CHANNELS, 0);
        warp_crop_into(frame, &Template::Ffhq512.align(landmarks, Enhancer::INPUT_SIZE), Enhancer::INPUT_SIZE, &mut input);
        let mut base = std::mem::take(&mut self.spare_base);
        base.resize(OUTPUT * OUTPUT * Frame::CHANNELS, 0);
        warp_crop_into(frame, &Template::Ffhq512.align(landmarks, OUTPUT), OUTPUT, &mut base);
        let job = Job { slot, generation, swap, input, base };
        self.busy = self.jobs.as_ref().is_some_and(|jobs| jobs.send(job).is_ok());
        Ok(())
    }

    /// Face `index` (at `generation`, its swap at version `swap`) of `frame` with its last
    /// restoration applied, or `None` while it has none. Hand its pixels back with
    /// [`AsyncEnhance::recycle`].
    pub fn crop(&mut self, index: usize, (generation, swap): (u64, u64), frame: &Frame, landmarks: &Landmarks5) -> Option<AlignedCrop> {
        let residual = self.residuals.get(index)?.as_ref().filter(|r| r.generation == generation)?;
        // Restored from this very swap, the residual fits the eyes and mouth too.
        let exact = residual.swap == swap;
        let to_crop = Template::Ffhq512.align(landmarks, OUTPUT);
        self.base.resize(OUTPUT * OUTPUT * Frame::CHANNELS, 0);
        warp_crop_into(frame, &to_crop, OUTPUT, &mut self.base);
        let unchanged = unchanged_cells(&residual.seen, &thumbnail(&self.base, OUTPUT));
        let mut pixels = std::mem::take(&mut self.pixels);
        pixels.resize(self.base.len(), 0);
        let row = OUTPUT * Frame::CHANNELS;
        let rows = pixels
            .par_chunks_mut(row)
            .zip(self.base.par_chunks(row))
            .zip(residual.values.par_chunks(row))
            .zip(self.weights.par_chunks(OUTPUT))
            .enumerate();
        let spans = &self.spans;
        rows.for_each(|(y, (((out, base), residual), weights))| {
            // The row's share of each column of cells, interpolated between cell rows.
            let mut gate = [0.0; THUMB];
            let (top, bottom, down) = spans[y];
            for (g, (t, b)) in gate.iter_mut().zip(unchanged[top * THUMB..][..THUMB].iter().zip(&unchanged[bottom * THUMB..][..THUMB])) {
                *g = down.mul_add(b - t, *t);
            }
            let pixels = out.as_chunks_mut::<3>().0.iter_mut().zip(base.as_chunks::<3>().0).zip(residual.as_chunks::<3>().0);
            for ((((o, b), r), &feature), &(left, right, across)) in pixels.zip(weights).zip(spans) {
                let w = if exact { 1.0 } else { feature } * across.mul_add(gate[right] - gate[left], gate[left]);
                for ((o, &b), &r) in o.iter_mut().zip(b).zip(r) {
                    *o = (f32::from(r).mul_add(w, f32::from(b))).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
        Some(AlignedCrop { pixels, size: OUTPUT, to_crop })
    }

    /// Give back a crop's pixel buffer for the next frame.
    pub fn recycle(&mut self, pixels: Vec<u8>) {
        self.pixels = pixels;
    }
}

impl Drop for AsyncEnhance {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Per thumbnail cell, how much of the residual still fits the face: 1 where it looks as the
/// restorer saw it, 0 where it changed.
fn unchanged_cells(seen: &Thumbnail, now: &Thumbnail) -> [f32; THUMB * THUMB] {
    let mut cells = [0.0; THUMB * THUMB];
    for ((cell, &a), &b) in cells.iter_mut().zip(seen).zip(now) {
        *cell = ((CHANGED_CELL - f32::from(a.abs_diff(b))) / (CHANGED_CELL - SAME_CELL)).clamp(0.0, 1.0);
    }
    cells
}

/// The two thumbnail cells around output pixel `i` (along one axis) and how far it lies from
/// the first towards the second, for bilinear interpolation between cell centres.
fn cell_span(i: usize) -> (usize, usize, f32) {
    let at = ((i as f32 + 0.5) * THUMB as f32 / OUTPUT as f32 - 0.5).clamp(0.0, (THUMB - 1) as f32);
    let first = at as usize;
    (first, (first + 1).min(THUMB - 1), at - first as f32)
}

/// 0 over the eyes and mouth of an aligned face, fading to 1 around them.
fn feature_weights() -> Vec<f32> {
    let [left_eye, right_eye, _, mouth_left, mouth_right] = Template::Ffhq512.points();
    let mouth = [f32::midpoint(mouth_left[0], mouth_right[0]), f32::midpoint(mouth_left[1], mouth_right[1])];
    let regions = [(left_eye, [EYE_RADIUS; 2]), (right_eye, [EYE_RADIUS; 2]), (mouth, MOUTH_RADII)];
    (0..OUTPUT * OUTPUT)
        .map(|i| {
            let (x, y) = ((i % OUTPUT) as f32 / OUTPUT as f32, (i / OUTPUT) as f32 / OUTPUT as f32);
            regions
                .iter()
                .map(|(center, [rx, ry])| {
                    // 1 at the edge of the ellipse, `1 + FEATHER` where the residual is whole again.
                    let distance = ((x - center[0]) / rx).hypot((y - center[1]) / ry);
                    ((distance - 1.0) / FEATHER).clamp(0.0, 1.0)
                })
                .fold(1.0, f32::min)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_residual_fades_out_where_the_face_changed() {
        let seen = [100; THUMB * THUMB];
        let mut now = seen;
        now[0] = 100 + CHANGED_CELL as u8;
        now[1] = 100 + SAME_CELL as u8;
        now[2] = 100 - (SAME_CELL + CHANGED_CELL) as u8 / 2;
        let cells = unchanged_cells(&seen, &now);
        assert_eq!(&cells[..4], &[0.0, 1.0, 0.5, 1.0]);
    }

    #[test]
    fn cells_are_interpolated_between_their_centres() {
        let cell = OUTPUT / THUMB;
        assert_eq!(cell_span(0), (0, 1, 0.0));
        assert_eq!(cell_span(cell / 2), (0, 1, (0.5 / cell as f32)));
        assert_eq!(cell_span(OUTPUT - 1), (THUMB - 1, THUMB - 1, 0.0));
    }

    #[test]
    fn eyes_and_mouth_are_left_out() {
        let weights = feature_weights();
        let at = |[x, y]: [f32; 2]| weights[(y * OUTPUT as f32) as usize * OUTPUT + (x * OUTPUT as f32) as usize];
        let [left_eye, right_eye, nose, mouth_left, _] = Template::Ffhq512.points();
        assert_eq!(at(left_eye), 0.0);
        assert_eq!(at(right_eye), 0.0);
        assert_eq!(at([0.5, 0.725]), 0.0);
        assert!(at(mouth_left) < 1.0);
        assert_eq!(at([0.5, 0.2]), 1.0, "forehead");
        assert_eq!(at([0.2, 0.6]), 1.0, "cheek");
        assert!(at(nose) > 0.0);
    }
}
