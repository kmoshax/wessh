use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use wajh_core::assets::SwapModel;
use wajh_core::{Device, Settings};
use wajh_media::CameraConfig;

#[derive(Debug, Parser)]
#[command(name = "wajh", version, about = "Real-time face swap. Run without a subcommand to open the app.")]
pub struct Cli {
    #[command(flatten)]
    pub options: Options,
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Options shared by every mode.
#[derive(Debug, Clone, Args)]
pub struct Options {
    /// ONNX Runtime library (libonnxruntime.so / onnxruntime.dll).
    #[arg(long, global = true)]
    pub ort: Option<PathBuf>,
    /// Inference backend.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub device: Device,
    /// Face swap model.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub model: SwapModel,
    /// Restore detail with GPEN after swapping.
    #[arg(long, global = true)]
    pub enhance: bool,
    /// Swap every face instead of only the largest.
    #[arg(long, global = true)]
    pub many_faces: bool,
    /// Detector input size; smaller is faster (320 suits close-up webcams).
    #[arg(long, global = true, default_value_t = 640)]
    pub detector_size: usize,
}

impl Options {
    pub fn settings(&self) -> Settings {
        Settings {
            swap_model: self.model,
            enhance: self.enhance,
            many_faces: self.many_faces,
            detector_size: self.detector_size,
            ..Settings::default()
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Swap faces in an image.
    Image {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Swap faces in a video file (audio is kept).
    Video {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: String,
        #[arg(short, long)]
        output: String,
        /// ffmpeg encoder (default: NVENC / `VideoToolbox` when available, else libx264).
        #[arg(long)]
        encoder: Option<String>,
    },
    /// Headless live mode: webcam → swap → virtual camera or stream.
    Cam {
        #[arg(short, long)]
        source: PathBuf,
        #[command(flatten)]
        camera: CameraArgs,
        /// /dev/videoN (v4l2loopback) or <udp://127.0.0.1:27000> (OBS media source).
        #[arg(long)]
        out: Option<String>,
    },
    /// Measure per-stage latency on an image.
    Bench {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short = 'n', long, default_value_t = 100)]
        iterations: usize,
    },
    /// Download the models the chosen options need.
    Models,
}

#[derive(Debug, Clone, Args)]
pub struct CameraArgs {
    /// Linux: /dev/video0, macOS: device index, Windows: video=<name>.
    #[arg(long, default_value_t = wajh_media::default_camera_device())]
    pub camera: String,
    #[arg(long, default_value_t = 1280)]
    pub width: usize,
    #[arg(long, default_value_t = 720)]
    pub height: usize,
    #[arg(long, default_value_t = 30)]
    pub fps: u32,
    /// Mirror the picture like a selfie camera.
    #[arg(long)]
    pub mirror: bool,
}

impl CameraArgs {
    pub fn config(&self) -> CameraConfig {
        CameraConfig { device: self.camera.clone(), width: self.width, height: self.height, fps: self.fps }
    }
}
