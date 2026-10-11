//! Webcam capture. Local cameras are read natively (V4L2, Media Foundation, AVFoundation through
//! `nokhwa`), so live mode needs no ffmpeg; network streams (phone camera apps) go through ffmpeg.

use crate::convert::{self, PixelFormat};
use crate::ffmpeg::{ffmpeg, spawn};
use crate::{LatestFrame, Result};
use nokhwa::utils::{ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use parking_lot::Mutex;
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

/// Formats we convert ourselves, most preferred first: MJPEG reaches 720p/1080p at 30 fps on
/// every UVC webcam, where uncompressed YUYV often tops out at a few frames per second.
const NATIVE_FORMATS: [FrameFormat; 6] =
    [FrameFormat::MJPEG, FrameFormat::YUYV, FrameFormat::NV12, FrameFormat::RAWRGB, FrameFormat::RAWBGR, FrameFormat::GRAY];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraConfig {
    /// Linux: `/dev/video0`; Windows and macOS: the camera's index (`0`); or a network stream
    /// such as a phone camera app (`http://192.168.1.20:8080/video`, `rtsp://…`).
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
    if cfg!(target_os = "linux") { "/dev/video0" } else { "0" }.to_owned()
}

/// The native camera index `device` names: `/dev/videoN` or a bare `N`.
fn native_index(device: &str) -> Option<u32> {
    device.strip_prefix("/dev/video").unwrap_or(device).parse().ok()
}

/// How frames are produced.
#[derive(Debug)]
enum Source {
    /// A local camera read on the capture thread.
    Native,
    /// ffmpeg pulling a network stream.
    Ffmpeg(Child),
}

/// Webcam capture on its own thread, publishing every frame into a [`LatestFrame`].
#[derive(Debug)]
pub struct Camera {
    source: Source,
    stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    device: String,
    last_error: Arc<Mutex<Option<String>>>,
}

impl Camera {
    pub fn start(config: &CameraConfig, output: Arc<LatestFrame>) -> Result<Self> {
        match native_index(&config.device) {
            Some(index) if !is_network_source(&config.device) => Ok(Self::start_native(index, config, output)?),
            _ => Self::start_ffmpeg(config, output),
        }
    }

    fn start_native(index: u32, config: &CameraConfig, output: Arc<LatestFrame>) -> Result<Self> {
        let (stop, last_error) = (Arc::new(AtomicBool::new(false)), Arc::new(Mutex::new(None)));
        let (stopped, failure, config_copy) = (stop.clone(), last_error.clone(), config.clone());
        let reader = std::thread::Builder::new().name("capture".into()).spawn(move || {
            if let Err(error) = capture_native(index, &config_copy, &output, &stopped) {
                *failure.lock() = Some(error);
            }
            // A clean stop leaves no error; anything else (unplugged camera) explains itself.
            if !stopped.load(Ordering::Relaxed) && failure.lock().is_none() {
                *failure.lock() = Some("no more frames".to_owned());
            }
        })?;
        Ok(Self { source: Source::Native, stop, reader: Some(reader), device: config.device.clone(), last_error })
    }

    fn start_ffmpeg(config: &CameraConfig, output: Arc<LatestFrame>) -> Result<Self> {
        let mut cmd = ffmpeg();
        cmd.args(input_args(config, std::env::consts::OS)).args(output_args(config)).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut process = spawn(&mut cmd)?;
        let mut stdout = process.stdout.take().expect("stdout is piped");
        let stderr = process.stderr.take().expect("stderr is piped");

        // Keep ffmpeg's last error line so a failure can be explained instead of just "no frames".
        let last_error = Arc::new(Mutex::new(None));
        let sink = last_error.clone();
        std::thread::Builder::new().name("capture-log".into()).spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(std::io::Result::ok) {
                if !line.trim().is_empty() {
                    *sink.lock() = Some(line);
                }
            }
        })?;

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
        Ok(Self { source: Source::Ffmpeg(process), stop, reader: Some(reader), device: config.device.clone(), last_error })
    }

    /// Why the camera stopped, once it has; `None` while it is still running.
    pub fn failure(&mut self) -> Option<String> {
        match &mut self.source {
            Source::Native => {
                self.reader.as_ref().is_some_and(JoinHandle::is_finished).then_some(())?;
            }
            Source::Ffmpeg(process) => {
                process.try_wait().ok().flatten()?;
            }
        }
        Some(explain_failure(&self.device, self.last_error.lock().as_deref()))
    }
}

/// Read camera `index` until `stop` is set, converting every frame to BGR at the configured size.
fn capture_native(index: u32, config: &CameraConfig, output: &LatestFrame, stop: &AtomicBool) -> std::result::Result<(), String> {
    request_camera_access();
    let wanted = CameraFormat::new_from(config.width as u32, config.height as u32, FrameFormat::MJPEG, config.fps);
    let request = RequestedFormat::with_formats(RequestedFormatType::Closest(wanted), &NATIVE_FORMATS);
    let mut camera = nokhwa::Camera::new(CameraIndex::Index(index), request).map_err(|e| e.to_string())?;
    camera.open_stream().map_err(|e| e.to_string())?;
    let (width, height) = (config.width, config.height);
    // The camera's own size when it can't deliver the configured one; letterboxed afterwards.
    let mut native = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let format = pixel_format(camera.frame_format());
        let resolution = camera.resolution();
        let (src_w, src_h) = (resolution.width() as usize, resolution.height() as usize);
        let raw = camera.frame_raw().map_err(|e| e.to_string())?;
        let mut frame = output.spare(width, height);
        if (src_w, src_h) == (width, height) {
            convert::to_bgr(format, &raw, width, height, &mut frame.data).map_err(|e| e.to_string())?;
        } else {
            native.resize(src_w * src_h * weshy_core::Frame::CHANNELS, 0);
            convert::to_bgr(format, &raw, src_w, src_h, &mut native).map_err(|e| e.to_string())?;
            convert::letterbox(&native, src_w, src_h, &mut frame.data, width, height);
        }
        output.publish(frame);
    }
    let _ = camera.stop_stream();
    Ok(())
}

fn pixel_format(format: FrameFormat) -> PixelFormat {
    match format {
        FrameFormat::MJPEG => PixelFormat::Mjpeg,
        FrameFormat::YUYV => PixelFormat::Yuyv,
        FrameFormat::NV12 => PixelFormat::Nv12,
        FrameFormat::RAWRGB => PixelFormat::Rgb,
        FrameFormat::RAWBGR => PixelFormat::Bgr,
        FrameFormat::GRAY => PixelFormat::Gray,
    }
}

/// macOS asks the user before an app may use the camera; wait for their answer once.
#[cfg(target_os = "macos")]
fn request_camera_access() {
    /// How long to wait for the user to answer the system's camera prompt.
    const PROMPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    if nokhwa::nokhwa_check() {
        return;
    }
    let (answered, answer) = std::sync::mpsc::channel();
    nokhwa::nokhwa_initialize(move |_granted| {
        let _ = answered.send(());
    });
    let _ = answer.recv_timeout(PROMPT_TIMEOUT);
}

#[cfg(not(target_os = "macos"))]
fn request_camera_access() {}

/// Local cameras as `(index, name)`, from the platform's native API.
pub(crate) fn native_cameras() -> Vec<(u32, String)> {
    request_camera_access();
    nokhwa::query(ApiBackend::Auto)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|info| match info.index() {
            CameraIndex::Index(index) => Some((*index, info.human_name())),
            CameraIndex::String(_) => None,
        })
        .collect()
}

/// A user-facing explanation of why capture from `device` failed, from ffmpeg's last error line.
fn explain_failure(device: &str, ffmpeg_error: Option<&str>) -> String {
    let detail = ffmpeg_error.unwrap_or("no error message");
    let has = |needle: &str| detail.to_ascii_lowercase().contains(needle);
    if is_network_source(device) {
        if has("connection refused") || has("timed out") || has("no route to host") || has("unreachable") {
            return format!(
                "Can't reach {device}. Make sure the camera app on your phone is open with its server started, \
                 the screen is on, and the phone is on the same Wi-Fi."
            );
        }
        if has("404") || has("not found") || has("invalid data") {
            return format!("{device} answered, but not with video. Paste the stream address the camera app shows.");
        }
    } else if has("busy") || has("in use") {
        return format!("{device} is in use by another app. Close it and start again.");
    } else if has("no such file") || has("no such device") || has("not found") || has("could not open") {
        return format!("Camera {device} was not found. Is it plugged in?");
    }
    format!("The camera stopped: {detail}")
}

/// Whether `device` names a network stream (phone camera apps, IP cameras) rather than a local device.
#[must_use]
pub fn is_network_source(device: &str) -> bool {
    device.contains("://")
}

/// Path of the MJPEG stream in IP Webcam (Android); DroidCam serves the same path.
const PHONE_STREAM_PATH: &str = "video";

/// The stream to open for `device`. A bare phone-app address (`http://192.168.1.3:8080/`) is its
/// control page, not video, so the app's stream path is appended.
fn stream_url(device: &str) -> String {
    let Some(rest) = device.strip_prefix("http://").or_else(|| device.strip_prefix("https://")) else {
        return device.to_owned();
    };
    let bare = !rest.trim_end_matches('/').contains('/');
    if bare { format!("{}/{PHONE_STREAM_PATH}", device.trim_end_matches('/')) } else { device.to_owned() }
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
        let url = stream_url(device);
        let mut args: Vec<String> = args.into_iter().map(str::to_owned).collect();
        args.extend(["-i".to_owned(), url]);
        return args;
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
        if let Source::Ffmpeg(process) = &mut self.source {
            let _ = process.kill();
            let _ = process.wait();
        }
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
    fn failures_are_explained() {
        let phone = "http://192.168.1.3:8080/";
        assert!(explain_failure(phone, Some("Connection to tcp://192.168.1.3:8080 failed: Connection refused")).contains("server started"));
        assert!(explain_failure("/dev/video0", Some("Device or resource busy")).contains("in use"));
        assert!(explain_failure("/dev/video9", Some("No such file or directory")).contains("not found"));
        assert!(explain_failure("/dev/video0", None).starts_with("The camera stopped"));
    }

    #[test]
    fn bare_phone_addresses_get_the_stream_path() {
        assert_eq!(stream_url("http://192.168.1.3:8080/"), "http://192.168.1.3:8080/video");
        assert_eq!(stream_url("http://192.168.1.3:8080"), "http://192.168.1.3:8080/video");
        assert_eq!(stream_url("http://192.168.1.3:4747/mjpegfeed"), "http://192.168.1.3:4747/mjpegfeed");
        assert_eq!(stream_url("rtsp://10.0.0.2:8554/live"), "rtsp://10.0.0.2:8554/live");
    }

    #[test]
    fn local_devices_are_read_natively() {
        assert_eq!(native_index("/dev/video2"), Some(2));
        assert_eq!(native_index("1"), Some(1));
        assert_eq!(native_index("http://192.168.1.3:8080/"), None);
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
