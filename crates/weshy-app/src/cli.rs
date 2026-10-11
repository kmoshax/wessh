use crate::install::{platform::Flavour, sources::ORT_VERSION};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use weshy_core::assets::{OccluderModel, SwapModel};
use weshy_core::face::{FaceRegion, FaceRegions};
use weshy_core::pipeline::FaceOrder;
use weshy_core::{Device, Settings};
use weshy_media::CameraConfig;

#[derive(Debug, Parser)]
#[command(name = "weshy", version, about = "Real-time face swap. Run without a subcommand to open the app.")]
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
    /// Face swap model. Developer / benchmarking option: weshy ships one curated model set, and
    /// setup and the app's onboarding only install the default model.
    #[arg(long, global = true, value_enum, default_value_t, hide = true)]
    pub model: SwapModel,
    /// Restore detail with GFPGAN after swapping, as much as the original face had: large, sharp
    /// faces get it, small or soft ones (most video) are left alone and cost nothing extra.
    #[arg(long, global = true, default_value_t = Settings::default().enhance, action = clap::ArgAction::Set)]
    pub enhance: bool,
    /// Swap every face instead of only the largest.
    #[arg(long, global = true)]
    pub many_faces: bool,
    /// Live modes: swap at most this many faces per frame; the others take turns and reuse their
    /// last swapped face meanwhile. 0 = no limit.
    #[arg(long, global = true, default_value_t = Settings::default().max_faces_per_frame)]
    pub max_faces_per_frame: usize,
    /// Detector input size; smaller is faster (320 suits close-up webcams).
    #[arg(long, global = true, default_value_t = 640)]
    pub detector_size: usize,
    /// Swapped face resolution: 256/384/512 run a 128 px model on sub-sampled tiles (much sharper,
    /// costs (size/128)² swaps per face; great for images and video). 0 = model's native size.
    #[arg(long, global = true, default_value_t = Settings::default().pixel_boost)]
    pub pixel_boost: usize,
    /// Keep hands, hair and glasses in front of the face (`XSeg` occlusion mask).
    #[arg(long, global = true, default_value_t = Settings::default().occlusion, action = clap::ArgAction::Set)]
    pub occlusion: bool,
    /// Occlusion model.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub occluder_model: OccluderModel,
    /// Only swap these face regions (`BiSeNet` parser), e.g. `skin,nose,eyes`.
    #[arg(long, global = true, value_enum, value_delimiter = ',')]
    pub face_regions: Vec<FaceRegion>,
    /// Keep the original mouth and lips (natural lip-sync when talking).
    #[arg(long, global = true)]
    pub mouth_mask: bool,
    /// Match the swapped face's colours to the original, 0 (off) to 1.
    #[arg(long, global = true, default_value_t = Settings::default().color_correction)]
    pub color_correction: f32,
    /// Order faces are picked in; without --many-faces the first one is swapped.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub face_order: FaceOrder,
    /// Only swap faces that look like a face in this image.
    #[arg(long, global = true)]
    pub reference: Option<PathBuf>,
    /// Which face of the reference image (position in --face-order, from 0).
    #[arg(long, global = true, default_value_t = 0)]
    pub reference_face: usize,
    /// Maximum identity distance (1 - cosine similarity) to the reference face.
    #[arg(long, global = true, default_value_t = Settings::default().reference_distance)]
    pub reference_distance: f32,
}

impl Options {
    pub fn settings(&self) -> Settings {
        Settings {
            swap_model: self.model,
            enhance: self.enhance,
            many_faces: self.many_faces,
            max_faces_per_frame: self.max_faces_per_frame,
            detector_size: self.detector_size,
            pixel_boost: self.pixel_boost,
            occlusion: self.occlusion,
            occluder_model: self.occluder_model,
            region_mask: !self.face_regions.is_empty(),
            face_regions: if self.face_regions.is_empty() { FaceRegions::ALL } else { self.face_regions.iter().copied().collect() },
            mouth_mask: self.mouth_mask,
            color_correction: self.color_correction,
            face_order: self.face_order,
            reference_distance: self.reference_distance,
            ..Settings::default()
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Swap faces in an image.
    Image {
        /// Source face image; repeat to average several photos of the same person.
        #[arg(short, long, required = true, num_args = 1..)]
        source: Vec<PathBuf>,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Swap faces in a video file (audio is kept).
    Video {
        /// Source face image; repeat to average several photos of the same person.
        #[arg(short, long, required = true, num_args = 1..)]
        source: Vec<PathBuf>,
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
        /// Source face image; repeat to average several photos of the same person.
        #[arg(short, long, required = true, num_args = 1..)]
        source: Vec<PathBuf>,
        #[command(flatten)]
        camera: CameraArgs,
        /// /dev/videoN (v4l2loopback) or <udp://127.0.0.1:27000> (OBS media source).
        #[arg(long)]
        out: Option<String>,
    },
    /// Measure per-stage latency on an image.
    Bench {
        /// Source face image; repeat to average several photos of the same person.
        #[arg(short, long, required = true, num_args = 1..)]
        source: Vec<PathBuf>,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short = 'n', long, default_value_t = 100)]
        iterations: usize,
        /// Also print per-stage p50/p95 and a CPU-kernel breakdown at 1080p.
        #[arg(long)]
        profile: bool,
        /// Move the camera (pan and zoom) from frame to frame, like handheld live video, instead
        /// of repeating one still frame — what the live-mode mask cache sees in practice.
        #[arg(long)]
        motion: bool,
    },
    /// Download the curated models (what the app's onboarding installs).
    Models,
    /// Install the ONNX Runtime build for this GPU, its CUDA (and TensorRT) libraries and the models.
    Setup(SetupArgs),
    /// Check the installation and explain how to fix what is missing.
    Doctor,
}

#[derive(Debug, Clone, Args)]
pub struct SetupArgs {
    /// Show what would be downloaded, with sizes, without changing anything.
    #[arg(long)]
    pub dry_run: bool,
    /// Runtime build to install [default: chosen from the GPU and driver].
    #[arg(long, value_enum)]
    pub flavour: Option<Flavour>,
    /// Skip TensorRT, which is otherwise installed on NVIDIA GPUs (a 3.5 GB download; the
    /// fastest backend). The app will offer to download it later.
    #[arg(long)]
    pub no_tensorrt: bool,
    /// ONNX Runtime release to install.
    #[arg(long, default_value = ORT_VERSION)]
    pub ort_version: String,
    /// Download and reinstall components even if they are already present.
    #[arg(long)]
    pub force: bool,
    /// Skip the models (the app downloads them on first start).
    #[arg(long)]
    pub no_models: bool,
}

#[derive(Debug, Clone, Args)]
pub struct CameraArgs {
    /// Linux: /dev/video0, macOS: device index, Windows: video=<name>.
    #[arg(long, default_value_t = weshy_media::default_camera_device())]
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
