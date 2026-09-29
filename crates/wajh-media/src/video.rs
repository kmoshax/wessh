use crate::ffmpeg::{ffmpeg, ffprobe, raw_bgr_input, spawn};
use crate::{Error, FrameSink, Result};
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};
use wajh_core::Frame;

#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frame_count: Option<u64>,
}

/// Read the first video stream's geometry and frame rate.
pub fn probe(path: &str) -> Result<VideoInfo> {
    let output = ffprobe()
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "default=nw=1", path])
        .output()
        .map_err(|source| Error::Spawn { program: "ffprobe".into(), source })?;
    let text = String::from_utf8_lossy(&output.stdout);
    let field = |key: &str| text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('=')).map(str::trim);
    let number = |key: &str| field(key).and_then(|v| v.parse::<usize>().ok()).ok_or_else(|| Error::NoVideoStream(path.to_owned()));
    let fps = field("avg_frame_rate")
        .and_then(|rate| {
            let (n, d) = rate.split_once('/')?;
            let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
            (d > 0.0).then(|| n / d)
        })
        .unwrap_or(30.0);
    Ok(VideoInfo { width: number("width")?, height: number("height")?, fps, frame_count: field("nb_frames").and_then(|n| n.parse().ok()) })
}

/// Decodes a video file into BGR frames.
#[derive(Debug)]
pub struct VideoReader {
    process: Child,
    stdout: ChildStdout,
    size: (usize, usize),
}

impl VideoReader {
    pub fn open(path: &str, info: &VideoInfo) -> Result<Self> {
        let mut cmd = ffmpeg();
        cmd.args(["-hwaccel", "auto", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]).stdout(Stdio::piped());
        let mut process = spawn(&mut cmd)?;
        let stdout = process.stdout.take().expect("stdout is piped");
        Ok(Self { process, stdout, size: (info.width, info.height) })
    }
}

impl Iterator for VideoReader {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        let mut frame = Frame::new(self.size.0, self.size.1);
        self.stdout.read_exact(&mut frame.data).ok().map(|()| frame)
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

/// H.264 encoder choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoder(pub String);

impl Encoder {
    /// Fastest available: NVENC, `VideoToolbox`, else libx264.
    #[must_use]
    pub fn best_available() -> Self {
        let listed = ffmpeg().arg("-encoders").stderr(Stdio::null()).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let nvidia = std::path::Path::new("/dev/nvidia0").exists() || cfg!(target_os = "windows");
        let name = if nvidia && listed.contains("h264_nvenc") {
            "h264_nvenc"
        } else if cfg!(target_os = "macos") && listed.contains("h264_videotoolbox") {
            "h264_videotoolbox"
        } else {
            "libx264"
        };
        Self(name.to_owned())
    }

    fn quality_args(&self, cmd: &mut Command) {
        match self.0.as_str() {
            "h264_nvenc" => cmd.args(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19"]),
            "libx264" => cmd.args(["-preset", "fast", "-crf", "18"]),
            _ => cmd.args(["-b:v", "12M"]),
        };
    }

    /// Open `output` for writing, muxing in the audio of `audio_source` untouched.
    pub fn open(&self, output: &str, audio_source: &str, info: &VideoInfo) -> Result<FrameSink> {
        let mut cmd = ffmpeg();
        cmd.arg("-y");
        raw_bgr_input(&mut cmd, info.width, info.height, info.fps);
        cmd.args(["-i", audio_source, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-c:v", &self.0, "-pix_fmt", "yuv420p"]);
        self.quality_args(&mut cmd);
        cmd.args(["-shortest", output]);
        FrameSink::spawn(cmd, info.width, info.height)
    }
}
