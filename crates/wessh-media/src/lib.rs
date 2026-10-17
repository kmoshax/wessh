//! Video I/O: native webcam capture and v4l2loopback output (live mode needs no ffmpeg), and
//! ffmpeg subprocesses for network streams and video files, which do all decoding and encoding
//! (incl. NVENC / `VideoToolbox`); only raw BGR frames cross the pipes.

mod camera;
mod convert;
mod devices;
mod error;
mod ffmpeg;
mod latest;
mod network;
mod sink;
mod video;

pub use camera::{Camera, CameraConfig, default_camera_device, is_network_source, restore_camera_settings};
pub use devices::{Devices, LOOPBACK_MODPROBE, LoopbackStatus, VideoDevice, list_devices};
pub use error::{Error, Result};
pub use ffmpeg::{bundled_ffmpeg, find_ffmpeg, system_ffmpeg};
pub use latest::LatestFrame;
pub use network::find_network_cameras;
pub use sink::{DEFAULT_LOOPBACK, DEFAULT_STREAM, FrameSink, OutputTarget};
pub use video::{Encoder, VideoInfo, VideoReader, probe};
