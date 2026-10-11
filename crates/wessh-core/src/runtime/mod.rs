//! ONNX Runtime integration: library discovery, execution-provider selection and model sessions.

mod binding;
mod engine;
pub mod locate;
mod model;
pub mod proto;

pub use binding::StaticModel;
pub use engine::{Backend, Device, Engine, StaticShapes};
pub use model::{Input, Model};
