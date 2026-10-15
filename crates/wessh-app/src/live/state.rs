use parking_lot::Mutex;
use wessh_core::{Settings, Stats};

/// RGBA pixels ready for upload to a texture.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// What the worker is doing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Loading,
    Ready,
    Live,
    Failed(String),
}

/// Everything the UI shows about the session, copied out under one lock.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub phase: Phase,
    /// Transient notice (e.g. "no face in source image").
    pub notice: Option<String>,
    pub backend: String,
    pub has_source: bool,
    pub source_name: Option<String>,
    /// The accepted source photo, for offline jobs to load the same face.
    pub source_path: Option<std::path::PathBuf>,
    pub stats: Stats,
    /// Frames processed per second.
    pub fps: f32,
    /// Frames the camera delivers per second.
    pub capture_fps: f32,
    /// From the camera delivering a frame to the frame being written out (virtual camera and
    /// preview), smoothed; the camera's exposure and the screen's refresh come on top.
    pub latency_ms: f32,
    pub output_active: bool,
}

pub(super) struct Shared {
    pub settings: Mutex<Settings>,
    pub snapshot: Mutex<Snapshot>,
    pub preview: Mutex<Option<Image>>,
    /// A preview pixel buffer the UI is done with, for the presenter to refill.
    pub recycled: Mutex<Option<Vec<u8>>>,
    pub source_thumbnail: Mutex<Option<Image>>,
}

impl Shared {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings: Mutex::new(settings),
            snapshot: Mutex::new(Snapshot::default()),
            preview: Mutex::new(None),
            recycled: Mutex::new(None),
            source_thumbnail: Mutex::new(None),
        }
    }

    pub fn update(&self, f: impl FnOnce(&mut Snapshot)) {
        f(&mut self.snapshot.lock());
    }
}
