//! Offline jobs (photos and video files) for the app, run one at a time on a worker thread
//! with its own [`wessh_core::Processor`], so they never share or block the live session.
//!
//! The engine and models load lazily on the first job and stay loaded for the next.

pub mod image;
pub mod video;
mod worker;

use crate::live::{EngineOptions, Image};
use image::SwappedImage;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use video::Progress;
use wessh_core::{Settings, Stats};
use wessh_media::{Encoder, VideoInfo};

/// Work for the runner.
#[derive(Clone, Debug)]
pub enum Job {
    Image { source: PathBuf, target: PathBuf, settings: Settings },
    Video { source: PathBuf, target: PathBuf, output: PathBuf, encoder: Encoder, info: VideoInfo, settings: Settings },
}

/// Where a job is.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Status {
    /// Waiting for an earlier job.
    #[default]
    Queued,
    /// Loading the engine, models or source face.
    Loading,
    /// Rendering a video.
    Running(Progress),
    Done(Finished),
    Failed(String),
    Cancelled,
}

impl Status {
    /// The job will not change any more.
    #[must_use]
    pub fn is_final(&self) -> bool {
        matches!(self, Self::Done(_) | Self::Failed(_) | Self::Cancelled)
    }
}

/// Summary of a finished job (the photo itself comes from [`JobHandle::take_image`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Finished {
    Image(Stats),
    Video { output: PathBuf, progress: Progress },
}

struct JobShared {
    status: Mutex<Status>,
    cancel: AtomicBool,
    image: Mutex<Option<SwappedImage>>,
    preview: Mutex<Option<Image>>,
}

/// The UI's view of one submitted job.
pub struct JobHandle {
    shared: Arc<JobShared>,
}

impl JobHandle {
    #[must_use]
    pub fn status(&self) -> Status {
        self.shared.status.lock().clone()
    }

    /// Ask the job to stop; a queued job is skipped.
    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    /// The swapped photo, once.
    #[must_use]
    pub fn take_image(&self) -> Option<SwappedImage> {
        self.shared.image.lock().take()
    }

    /// The newest processed video frame, if one arrived since the last call.
    #[must_use]
    pub fn take_preview(&self) -> Option<Image> {
        self.shared.preview.lock().take()
    }
}

struct Queued {
    job: Job,
    shared: Arc<JobShared>,
}

/// Handle to the job worker thread. Dropping it cancels every job and waits for the worker,
/// so GPU resources are released before the process exits (CUDA aborts otherwise).
pub struct JobRunner {
    queue: Option<Sender<Queued>>,
    worker: Option<JoinHandle<()>>,
    /// Jobs that may still be queued or running, to cancel on drop.
    submitted: Mutex<Vec<Weak<JobShared>>>,
}

impl JobRunner {
    /// Updating must not interrupt queued or running image/video processing.
    pub fn is_busy(&self) -> bool {
        self.submitted
            .lock()
            .iter()
            .filter_map(Weak::upgrade)
            .any(|job| matches!(*job.status.lock(), Status::Queued | Status::Loading | Status::Running(_)))
    }
    /// Spawn the (idle) worker. `repaint` is called whenever a job has news.
    pub fn spawn(engine: EngineOptions, repaint: impl Fn() + Send + 'static) -> Self {
        let (queue, inbox) = std::sync::mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("jobs".into())
            .spawn(move || worker::run(&engine, &inbox, &repaint))
            .expect("spawning the job worker thread");
        Self { queue: Some(queue), worker: Some(worker), submitted: Mutex::default() }
    }

    /// Queue `job`; it runs after the jobs submitted before it.
    #[must_use]
    pub fn submit(&self, job: Job) -> JobHandle {
        let shared = Arc::new(JobShared {
            status: Mutex::new(Status::Queued),
            cancel: AtomicBool::new(false),
            image: Mutex::new(None),
            preview: Mutex::new(None),
        });
        let mut submitted = self.submitted.lock();
        submitted.retain(|job| job.strong_count() > 0);
        submitted.push(Arc::downgrade(&shared));
        let queued = Queued { job, shared: shared.clone() };
        if self.queue.as_ref().is_none_or(|queue| queue.send(queued).is_err()) {
            *shared.status.lock() = Status::Failed("the job worker stopped".into());
        }
        JobHandle { shared }
    }
}

impl Drop for JobRunner {
    fn drop(&mut self) {
        for job in self.submitted.lock().drain(..).filter_map(|job| job.upgrade()) {
            job.cancel.store(true, Ordering::Relaxed);
        }
        drop(self.queue.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
