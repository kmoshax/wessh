//! Video I/O through ffmpeg subprocesses: webcam capture, virtual-camera / stream output and
//! video files. ffmpeg does all decoding and encoding (incl. NVENC / `VideoToolbox`) natively;
//! only raw BGR frames cross the pipes.

mod camera;
mod devices;
mod error;
mod ffmpeg;
mod latest;
mod sink;
mod video;

pub use camera::{Camera, CameraConfig, default_camera_device};
pub use devices::{Devices, LOOPBACK_MODPROBE, LoopbackStatus, VideoDevice, list_devices};
pub use error::{Error, Result};
pub use latest::LatestFrame;
pub use sink::{DEFAULT_LOOPBACK, DEFAULT_STREAM, FrameSink, OutputTarget};
pub use video::{Encoder, VideoInfo, VideoReader, probe};
