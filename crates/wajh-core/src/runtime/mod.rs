//! ONNX Runtime integration: library discovery, execution-provider selection and model sessions.

mod binding;
mod engine;
mod locate;
mod model;
pub mod proto;

pub use binding::StaticModel;
pub use engine::{Backend, Device, Engine};
pub use model::{Input, Model};
