//! Face swap engine: detection, identity embedding, swapping, enhancement and temporal
//! stabilisation, running on ONNX Runtime.

pub mod assets;
pub mod error;
pub mod face;
pub mod geometry;
pub mod image;
pub mod pipeline;
pub mod process;
pub mod runtime;

pub use error::{Error, Result};
pub use image::Frame;
pub use pipeline::{Processor, Settings, Stats};
pub use runtime::{Backend, Device, Engine};
