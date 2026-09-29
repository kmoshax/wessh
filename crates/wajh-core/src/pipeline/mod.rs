//! The per-frame face swap pipeline.

pub mod pixel_boost;
mod processor;
mod reuse;
mod selection;
mod settings;
mod smoothing;
mod stats;
mod tracker;

pub use processor::Processor;
pub use selection::{FaceOrder, select};
pub use settings::Settings;
pub use smoothing::FaceSmoother;
pub use stats::Stats;
pub use tracker::{FaceSource, FaceTracker, Pass, TrackerConfig, TrackingDetector};
