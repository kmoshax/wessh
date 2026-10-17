//! Async timewarp for live video: the swap network runs on its own thread, beside the frame
//! loop, so a slow swap no longer holds back the picture.
//!
//! Every frame is still detected and every face pasted at camera rate, through the face's
//! *current* alignment ([`super::reuse`]); only the crop pasted comes from the newest swap that
//! has finished. The face follows the head without delay and the expression trails by one
//! swap. While a swap is in flight the next frames reuse the last one; when it lands, the
//! following frame picks it up and the next face's crop is handed over.

use super::pixel_boost;
use crate::face::Swapper;
use crate::geometry::Affine;
use crate::{Error, Result};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The swap network, shared by the frame loop (synchronous swaps) and the swap thread.
pub(super) type SharedSwapper = Arc<Mutex<Option<Swapper>>>;

/// Lock the swapper; a panic on the other side leaves it usable (each run starts afresh).
pub(super) fn lock(swapper: &SharedSwapper) -> MutexGuard<'_, Option<Swapper>> {
    swapper.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One swap handed to the thread.
struct Job {
    slot: usize,
    generation: u64,
    original: Vec<u8>,
    to_crop: Affine,
    size: usize,
    condition: Arc<[f32]>,
    taken: Instant,
}

/// A finished swap.
pub(super) struct Done {
    /// The face it belongs to and its generation then (see [`super::reuse::SwapReuse`]).
    pub slot: usize,
    pub generation: u64,
    pub size: usize,
    /// When the frame it was swapped from was captured.
    pub taken: Instant,
    /// How long the swap network ran.
    pub took: Duration,
    pub pixels: Result<Vec<u8>>,
    /// The aligned crop it was swapped from (hand it back with [`AsyncSwap::recycle`]) and its
    /// alignment.
    pub original: Vec<u8>,
    pub to_crop: Affine,
}

/// The swap thread, with at most one swap in flight (the GPU runs one network at a time anyway).
pub(super) struct AsyncSwap {
    jobs: Option<Sender<Job>>,
    done: Receiver<Done>,
    thread: Option<JoinHandle<()>>,
    busy: bool,
    /// A crop buffer to reuse for the next job.
    spare: Vec<u8>,
    /// How long the last swap took.
    last: Option<Duration>,
}

impl AsyncSwap {
    pub fn spawn(swapper: SharedSwapper) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (finished, done) = mpsc::channel();
        let thread = std::thread::Builder::new().name("swap".into()).spawn(move || {
            for job in inbox {
                let started = Instant::now();
                let pixels = match lock(&swapper).as_mut() {
                    Some(swapper) => pixel_boost::swap_aligned(swapper, &job.original, job.size, &job.condition),
                    None => Err(Error::UnexpectedOutput { model: "swapper", detail: "no swap model loaded".to_owned() }),
                };
                let took = started.elapsed();
                let Job { slot, generation, original, to_crop, size, taken, .. } = job;
                if finished.send(Done { slot, generation, size, taken, took, pixels, original, to_crop }).is_err() {
                    return;
                }
            }
        })?;
        Ok(Self { jobs: Some(jobs), done, thread: Some(thread), busy: false, spare: Vec::new(), last: None })
    }

    /// A swap is in flight.
    pub const fn is_busy(&self) -> bool {
        self.busy
    }

    /// A buffer for the next job's crop.
    pub fn buffer(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.spare)
    }

    /// Give back a crop buffer for reuse.
    pub fn recycle(&mut self, buffer: Vec<u8>) {
        self.spare = buffer;
    }

    /// Start swapping face `slot` (at `generation`) from its `size`² crop aligned by `to_crop`.
    pub fn submit(&mut self, slot: usize, generation: u64, condition: &Arc<[f32]>, original: Vec<u8>, to_crop: Affine, size: usize) {
        let job = Job { slot, generation, original, to_crop, size, condition: condition.clone(), taken: Instant::now() };
        self.busy = self.jobs.as_ref().is_some_and(|jobs| jobs.send(job).is_ok());
    }

    /// The swap that finished since the last call, if any.
    pub fn take(&mut self) -> Option<Done> {
        match self.done.try_recv() {
            Ok(done) => Some(self.landed(done)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.busy = false;
                None
            }
        }
    }

    /// Wait up to `budget` for the swap in flight, when swaps have lately been that quick; `None`
    /// when it didn't land in time (it lands on a later frame instead).
    pub fn wait(&mut self, budget: Duration) -> Option<Done> {
        if !self.busy || self.last.is_none_or(|last| last > budget) {
            return None;
        }
        match self.done.recv_timeout(budget) {
            Ok(done) => Some(self.landed(done)),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => {
                self.busy = false;
                None
            }
        }
    }

    fn landed(&mut self, done: Done) -> Done {
        self.busy = false;
        self.last = Some(done.took);
        done
    }
}

impl Drop for AsyncSwap {
    /// Close the job queue and wait for the swap in flight, so the model outlives its last run.
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
