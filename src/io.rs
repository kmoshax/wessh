//! Video I/O through ffmpeg subprocesses: webcam capture, virtual camera / stream output and
//! video files. ffmpeg does the MJPEG/H.264 decoding and encoding (incl. NVENC) in native code
//! and we only move raw BGR frames through pipes.

use crate::imgops::Frame;
use anyhow::{Context, Result, bail};
use parking_lot::{Condvar, Mutex};
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn ffmpeg() -> Command {
    let mut c = Command::new(std::env::var("WAJH_FFMPEG").unwrap_or_else(|_| "ffmpeg".into()));
    c.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    c
}

/// Holds only the newest frame: readers never see stale frames, so latency stays bounded.
#[derive(Default)]
pub struct LatestFrame {
    slot: Mutex<(Option<Frame>, u64)>,
    cv: Condvar,
}

impl LatestFrame {
    pub fn put(&self, f: Frame) {
        let mut s = self.slot.lock();
        s.0 = Some(f);
        s.1 += 1;
        self.cv.notify_all();
    }

    /// Wait (up to `timeout`) for a frame newer than `last_seq`.
    pub fn take_newer(&self, last_seq: &mut u64, timeout: std::time::Duration) -> Option<Frame> {
        let mut s = self.slot.lock();
        if s.1 == *last_seq {
            self.cv.wait_for(&mut s, timeout);
        }
        if s.1 == *last_seq {
            return None;
        }
        *last_seq = s.1;
        s.0.take()
    }
}

pub struct CameraConfig {
    pub device: String,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
}

pub fn default_camera_device() -> String {
    if cfg!(target_os = "linux") {
        "/dev/video0".into()
    } else if cfg!(target_os = "macos") {
        "0".into()
    } else {
        "video=Integrated Camera".into()
    }
}

/// Webcam capture running on its own thread and publishing into a [`LatestFrame`].
pub struct Camera {
    child: Child,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Camera {
    pub fn start(cfg: &CameraConfig, out: Arc<LatestFrame>) -> Result<Self> {
        let size = format!("{}x{}", cfg.width, cfg.height);
        let fps = cfg.fps.to_string();
        let mut cmd = ffmpeg();
        cmd.args(["-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0"]);
        if cfg!(target_os = "linux") {
            cmd.args(["-f", "v4l2", "-input_format", "mjpeg", "-framerate", &fps, "-video_size", &size, "-i", &cfg.device]);
        } else if cfg!(target_os = "macos") {
            cmd.args(["-f", "avfoundation", "-framerate", &fps, "-video_size", &size, "-i", &cfg.device]);
        } else {
            cmd.args(["-f", "dshow", "-rtbufsize", "16M", "-framerate", &fps, "-video_size", &size, "-i", &cfg.device]);
        }
        cmd.args(["-vf", &format!("scale={}:{}", cfg.width, cfg.height), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]);
        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("starting ffmpeg (is it installed?)")?;
        let mut stdout = child.stdout.take().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (w, h) = (cfg.width, cfg.height);
        let stop2 = stop.clone();
        let thread = std::thread::Builder::new().name("capture".into()).spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                let mut f = Frame::new(w, h);
                if stdout.read_exact(&mut f.data).is_err() {
                    break;
                }
                out.put(f);
            }
        })?;
        Ok(Self { child, stop, thread: Some(thread) })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Raw-frame sink: virtual camera (v4l2loopback), network stream for OBS, or anything ffmpeg can write.
pub struct FrameSink {
    child: Child,
    stdin: Option<ChildStdin>,
    width: usize,
    height: usize,
}

impl FrameSink {
    /// `target` examples: `/dev/video10` (v4l2loopback), `udp://127.0.0.1:27000` (OBS media source).
    pub fn virtual_camera(target: &str, width: usize, height: usize, fps: u32) -> Result<Self> {
        let mut cmd = ffmpeg();
        raw_input(&mut cmd, width, height, fps as f64);
        if target.starts_with("udp://") || target.starts_with("tcp://") || target.starts_with("srt://") {
            cmd.args(["-c:v", "mpeg1video", "-q:v", "3", "-bf", "0", "-f", "mpegts", target]);
        } else if cfg!(target_os = "linux") {
            cmd.args(["-f", "v4l2", "-pix_fmt", "yuyv422", target]);
        } else {
            bail!("native virtual camera output is Linux-only for now; use a udp:// target with OBS");
        }
        Self::spawn(cmd, width, height)
    }

    fn spawn(mut cmd: Command, width: usize, height: usize) -> Result<Self> {
        let mut child = cmd.stdin(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("starting ffmpeg")?;
        let stdin = child.stdin.take();
        Ok(Self { child, stdin, width, height })
    }

    pub fn write(&mut self, f: &Frame) -> Result<()> {
        if f.width != self.width || f.height != self.height {
            bail!("frame size changed");
        }
        self.stdin.as_mut().context("sink closed")?.write_all(&f.data)?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        if !status.success() {
            bail!("ffmpeg exited with {status}");
        }
        Ok(())
    }
}

impl Drop for FrameSink {
    fn drop(&mut self) {
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

fn raw_input(cmd: &mut Command, w: usize, h: usize, fps: f64) {
    cmd.args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", &format!("{w}x{h}"), "-r", &format!("{fps}"), "-i", "-"]);
}

pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames: Option<u64>,
}

pub fn probe(path: &str) -> Result<VideoInfo> {
    let out = Command::new(std::env::var("WAJH_FFPROBE").unwrap_or_else(|_| "ffprobe".into()))
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "default=nw=1", path])
        .output()
        .context("running ffprobe")?;
    let text = String::from_utf8_lossy(&out.stdout);
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).map(str::trim);
    let width = get("width").context("no video stream")?.parse()?;
    let height = get("height").context("no video stream")?.parse()?;
    let fps = get("avg_frame_rate")
        .and_then(|r| {
            let (n, d) = r.split_once('/')?;
            let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
            (d > 0.0).then_some(n / d)
        })
        .unwrap_or(30.0);
    let frames = get("nb_frames").and_then(|n| n.parse().ok());
    Ok(VideoInfo { width, height, fps, frames })
}

pub struct VideoReader {
    child: Child,
    stdout: std::process::ChildStdout,
    width: usize,
    height: usize,
}

impl VideoReader {
    pub fn open(path: &str, info: &VideoInfo) -> Result<Self> {
        let mut cmd = ffmpeg();
        cmd.args(["-hwaccel", "auto", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]);
        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("starting ffmpeg")?;
        let stdout = child.stdout.take().unwrap();
        Ok(Self { child, stdout, width: info.width, height: info.height })
    }

    pub fn next(&mut self) -> Option<Frame> {
        let mut f = Frame::new(self.width, self.height);
        self.stdout.read_exact(&mut f.data).ok().map(|_| f)
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Pick the fastest available H.264 encoder.
pub fn best_encoder() -> &'static str {
    let listed = Command::new("ffmpeg").args(["-hide_banner", "-encoders"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let has_nvidia = std::path::Path::new("/dev/nvidia0").exists() || cfg!(target_os = "windows");
    if has_nvidia && listed.contains("h264_nvenc") {
        "h264_nvenc"
    } else if cfg!(target_os = "macos") && listed.contains("h264_videotoolbox") {
        "h264_videotoolbox"
    } else {
        "libx264"
    }
}

/// Encoder that also copies the audio track from `audio_from`.
pub fn video_writer(out: &str, audio_from: &str, info: &VideoInfo, encoder: &str) -> Result<FrameSink> {
    let mut cmd = ffmpeg();
    cmd.arg("-y");
    raw_input(&mut cmd, info.width, info.height, info.fps);
    cmd.args(["-i", audio_from, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-c:v", encoder, "-pix_fmt", "yuv420p"]);
    match encoder {
        "h264_nvenc" => cmd.args(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19"]),
        "libx264" => cmd.args(["-preset", "fast", "-crf", "18"]),
        _ => cmd.args(["-b:v", "12M"]),
    };
    cmd.args(["-shortest", out]);
    FrameSink::spawn(cmd, info.width, info.height)
}
