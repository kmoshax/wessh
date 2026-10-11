use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ONNX Runtime library not found; place it next to the binary, in {searched}, or set ORT_DYLIB_PATH")]
    RuntimeNotFound { searched: PathBuf },

    #[error("failed to load ONNX Runtime from {path}: {reason}")]
    RuntimeLoad { path: PathBuf, reason: String },

    #[error("failed to load model {path}: {source}")]
    ModelLoad { path: PathBuf, source: ort::Error },

    #[error("inference failed: {0}")]
    Inference(#[from] ort::Error),

    #[error("model {model} returned unexpected output: {detail}")]
    UnexpectedOutput { model: &'static str, detail: String },

    #[error("model {model} has a dynamic shape: {detail}")]
    DynamicShape { model: &'static str, detail: String },

    #[error("download of {url} failed: {reason}")]
    Download { url: String, reason: String },

    #[error("invalid ONNX file: {0}")]
    InvalidOnnx(String),

    #[error("no face found in the source image")]
    NoSourceFace,

    #[error("reference face #{index} requested but the reference image has {found} face(s)")]
    NoReferenceFace { index: usize, found: usize },

    #[error("image error: {0}")]
    Image(#[from] ::image::ImageError),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
