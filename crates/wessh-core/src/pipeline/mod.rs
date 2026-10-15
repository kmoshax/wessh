//! The per-frame face swap pipeline.

mod async_enhance;
mod async_swap;
mod flow;
pub mod pixel_boost;
mod processor;
mod reuse;
mod selection;
mod settings;
mod stats;
mod tracker;

pub use processor::Processor;
pub use selection::{FaceOrder, select};
pub use settings::Settings;
pub use stats::Stats;
pub use tracker::{FaceSource, FaceTracker, Pass, TrackerConfig, TrackingDetector};
