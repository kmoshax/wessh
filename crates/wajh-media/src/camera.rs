use crate::ffmpeg::{ffmpeg, spawn};
use crate::{LatestFrame, Result};
use std::io::Read;
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraConfig {
    /// Linux: `/dev/video0`; macOS: device index; Windows: `video=<device name>`; or a network
    /// stream such as a phone camera app (`http://192.168.1.20:8080/video`, `rtsp://…`).
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
        let mut cmd = ffmpeg();
        cmd.args(input_args(config, std::env::consts::OS)).args(output_args(config)).stdout(Stdio::piped());
        let mut process = spawn(&mut cmd)?;
        let mut stdout = process.stdout.take().expect("stdout is piped");

        let stop = Arc::new(AtomicBool::new(false));
        let (width, height, stopped) = (config.width, config.height, stop.clone());
        let reader = std::thread::Builder::new().name("capture".into()).spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let mut frame = output.spare(width, height);
                if stdout.read_exact(&mut frame.data).is_err() {
                    break;
                }
                output.publish(frame);
            }
        })?;
        Ok(Self { process, stop, reader: Some(reader) })
    }
}

/// Whether `device` names a network stream (phone camera apps, IP cameras) rather than a local device.
#[must_use]
pub fn is_network_source(device: &str) -> bool {
    device.contains("://")
}

/// ffmpeg arguments that open the camera, tuned for the lowest latency.
fn input_args(config: &CameraConfig, os: &str) -> Vec<String> {
    let mut args: Vec<&str> = vec!["-fflags", "nobuffer", "-flags", "low_delay"];
    let device = config.device.as_str();
    let (size, fps) = (format!("{}x{}", config.width, config.height), config.fps.to_string());
    if is_network_source(device) {
        if device.starts_with("rtsp://") {
            // UDP drops packets on Wi-Fi and smears frames; TCP keeps them intact.
            args.extend(["-rtsp_transport", "tcp"]);
        } else if device.starts_with("http") {
            args.extend(["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]);
        }
        args.extend(["-i", device]);
        return args.into_iter().map(str::to_owned).collect();
    }
    // Local devices: skip stream probing, it only adds startup delay.
    args.extend(["-probesize", "32", "-analyzeduration", "0"]);
    match os {
        "linux" => args.extend(["-f", "v4l2", "-input_format", "mjpeg"]),
        "macos" => args.extend(["-f", "avfoundation"]),
        _ => args.extend(["-f", "dshow", "-rtbufsize", "16M"]),
    }
    let mut args: Vec<String> = args.into_iter().map(str::to_owned).collect();
    args.extend(["-framerate".into(), fps, "-video_size".into(), size, "-i".into(), device.to_owned()]);
    args
}

/// Raw BGR frames at exactly the configured size. Sources with another aspect ratio (a phone
/// held upright) are letterboxed rather than stretched.
fn output_args(config: &CameraConfig) -> Vec<String> {
    let (w, h) = (config.width, config.height);
    let fit = format!("scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2");
    ["-vf", &fit, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"].map(str::to_owned).to_vec()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(device: &str) -> CameraConfig {
        CameraConfig { device: device.to_owned(), ..CameraConfig::default() }
    }

    #[test]
    fn phone_streams_are_opened_as_network_inputs() {
        let http = input_args(&config("http://192.168.1.20:8080/video"), "linux");
        assert!(http.contains(&"-reconnect".to_owned()));
        assert!(!http.contains(&"v4l2".to_owned()));
        let rtsp = input_args(&config("rtsp://10.0.0.2:8554/live"), "linux");
        assert!(rtsp.windows(2).any(|w| w == ["-rtsp_transport", "tcp"]));
    }

    #[test]
    fn local_devices_use_the_platform_grabber() {
        let args = input_args(&config("/dev/video0"), "linux");
        assert!(args.windows(2).any(|w| w == ["-f", "v4l2"]));
        assert_eq!(args.last().map(String::as_str), Some("/dev/video0"));
        assert!(input_args(&config("0"), "macos").contains(&"avfoundation".to_owned()));
    }

    #[test]
    fn output_is_letterboxed_to_the_configured_size() {
        let args = output_args(&CameraConfig { width: 1280, height: 720, ..config("x") });
        assert!(args[1].starts_with("scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720"));
    }
}
