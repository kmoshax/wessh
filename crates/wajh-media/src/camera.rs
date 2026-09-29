use crate::ffmpeg::{ffmpeg, spawn};
use crate::{LatestFrame, Result};
use std::io::Read;
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use wajh_core::Frame;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraConfig {
    /// Linux: `/dev/video0`; macOS: device index; Windows: `video=<device name>`.
    pub device: String,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self { device: default_camera_device(), width: 1280, height: 720, fps: 30 }
    }
}

#[must_use]
pub fn default_camera_device() -> String {
    if cfg!(target_os = "linux") {
        "/dev/video0"
    } else if cfg!(target_os = "macos") {
        "0"
    } else {
        "video=Integrated Camera"
    }
    .to_owned()
}

/// Webcam capture on its own thread, publishing every frame into a [`LatestFrame`].
#[derive(Debug)]
pub struct Camera {
    process: Child,
    stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
}

impl Camera {
    pub fn start(config: &CameraConfig, output: Arc<LatestFrame>) -> Result<Self> {
        let size = format!("{}x{}", config.width, config.height);
        let fps = config.fps.to_string();
        let mut cmd = ffmpeg();
        cmd.args(["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0"]);
        match std::env::consts::OS {
            "linux" => cmd.args(["-f", "v4l2", "-input_format", "mjpeg"]),
            "macos" => cmd.args(["-f", "avfoundation"]),
            _ => cmd.args(["-f", "dshow", "-rtbufsize", "16M"]),
        };
        cmd.args(["-framerate", &fps, "-video_size", &size, "-i", &config.device])
            .args(["-vf", &format!("scale={}:{}", config.width, config.height), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
            .stdout(Stdio::piped());
        let mut process = spawn(&mut cmd)?;
        let mut stdout = process.stdout.take().expect("stdout is piped");

        let stop = Arc::new(AtomicBool::new(false));
        let (width, height, stopped) = (config.width, config.height, stop.clone());
        let reader = std::thread::Builder::new().name("capture".into()).spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let mut frame = Frame::new(width, height);
                if stdout.read_exact(&mut frame.data).is_err() {
                    break;
                }
                output.publish(frame);
            }
        })?;
        Ok(Self { process, stop, reader: Some(reader) })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.process.kill();
        let _ = self.process.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
