//! The per-frame face swap pipeline.

pub mod pixel_boost;
mod processor;
mod selection;
mod settings;
mod smoothing;
mod stats;

pub use processor::Processor;
pub use selection::{FaceOrder, select};
pub use settings::Settings;
pub use smoothing::FaceSmoother;
pub use stats::Stats;
