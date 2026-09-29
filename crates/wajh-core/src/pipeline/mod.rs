//! The per-frame face swap pipeline.

mod processor;
mod settings;
mod smoothing;
mod stats;
mod tracker;

pub use processor::Processor;
pub use settings::Settings;
pub use smoothing::FaceSmoother;
pub use stats::Stats;
pub use tracker::{FaceSource, FaceTracker, Pass, TrackerConfig, TrackingDetector};
