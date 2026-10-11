//! Live session: capture → swap on a dedicated worker thread, preview conversion and virtual
//! camera output on a presenter thread.
//! The UI drives it only through [`LiveSession`]'s methods and reads [`Snapshot`]s back.

mod presenter;
mod state;
mod worker;

pub use state::{Image, Phase, Snapshot};

use crossbeam_channel::Sender;
use state::Shared;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use wessh_core::{Device, Settings};
use wessh_media::{CameraConfig, OutputTarget};

/// How to initialise inference.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub runtime: Option<PathBuf>,
    pub device: Device,
}

enum Command {
    SetSource(PathBuf),
    Start { camera: CameraConfig, output: Option<OutputTarget> },
    Stop,
}

/// Handle to the worker thread. Dropping it shuts the worker down and waits for it, so GPU
/// resources are released before the process exits (CUDA aborts otherwise).
pub struct LiveSession {
    commands: Sender<Command>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        // Disconnect the command channel so the worker's loop ends, then wait for it.
        drop(std::mem::replace(&mut self.commands, crossbeam_channel::unbounded().0));
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl LiveSession {
    /// Spawn the worker. `repaint` is called whenever there is something new to show.
    pub fn spawn(engine: EngineOptions, settings: Settings, repaint: impl Fn() + Send + Sync + 'static) -> Self {
        let (commands, inbox) = crossbeam_channel::unbounded();
        let shared = Arc::new(Shared::new(settings));
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("live".into())
            .spawn(move || worker::run(&engine, &worker_shared, &inbox, &(Arc::new(repaint) as Arc<dyn Fn() + Send + Sync>)))
            .expect("spawning the live worker thread");
        Self { commands, shared, worker: Some(worker) }
    }

    pub fn set_source(&self, path: PathBuf) {
        let _ = self.commands.send(Command::SetSource(path));
    }

    pub fn start(&self, camera: CameraConfig, output: Option<OutputTarget>) {
        let _ = self.commands.send(Command::Start { camera, output });
    }

    pub fn stop(&self) {
        let _ = self.commands.send(Command::Stop);
    }

    #[must_use]
    pub fn settings(&self) -> Settings {
        self.shared.settings.lock().clone()
    }

    pub fn set_settings(&self, settings: Settings) {
        *self.shared.settings.lock() = settings;
    }

    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        self.shared.snapshot.lock().clone()
    }

    /// The newest processed frame, if one arrived since the last call.
    #[must_use]
    pub fn take_preview(&self) -> Option<Image> {
        self.shared.preview.lock().take()
    }

    /// Hand back a preview's pixel buffer once it has been uploaded, so the next preview reuses it.
    pub fn recycle_preview(&self, image: Image) {
        *self.shared.recycled.lock() = Some(image.rgba);
    }

    /// Thumbnail of a newly loaded source face.
    #[must_use]
    pub fn take_source_thumbnail(&self) -> Option<Image> {
        self.shared.source_thumbnail.lock().take()
    }
}
