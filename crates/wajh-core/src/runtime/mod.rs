//! ONNX Runtime integration: library discovery, execution-provider selection and model sessions.

mod engine;
pub mod locate;
mod model;
pub mod proto;

pub use engine::{Backend, Device, Engine};
pub use model::{Input, Model};
