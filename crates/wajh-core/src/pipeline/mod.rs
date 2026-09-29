//! The per-frame face swap pipeline.

mod processor;
mod settings;
mod smoothing;
mod stats;

pub use processor::Processor;
pub use settings::Settings;
pub use smoothing::FaceSmoother;
pub use stats::Stats;
