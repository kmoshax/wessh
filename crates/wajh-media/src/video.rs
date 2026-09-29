use crate::ffmpeg::{ffmpeg, ffprobe, raw_bgr_input, spawn};
use crate::{Error, FrameSink, Result};
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};
use wajh_core::Frame;

/// Frame rate assumed when a file does not state one.
const FALLBACK_FPS: f64 = 30.0;

#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    /// Frame count from the container, when it records one.
    pub frame_count: Option<u64>,
    /// Container duration in seconds.
    pub duration: Option<f64>,
}

impl VideoInfo {
    /// The recorded frame count, else one estimated from the duration.
    #[must_use]
    pub fn total_frames(&self) -> Option<u64> {
        self.frame_count.or_else(|| self.duration.map(|seconds| (seconds * self.fps).round() as u64))
    }

    /// The duration, else one derived from the frame count.
    #[must_use]
    pub fn seconds(&self) -> Option<f64> {
        self.duration.or_else(|| self.frame_count.map(|frames| frames as f64 / self.fps))
    }
}

/// Read the first video stream's geometry and frame rate.
pub fn probe(path: &str) -> Result<VideoInfo> {
    let output = ffprobe()
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "default=nw=1",
            path,
        ])
        .output()
        .map_err(|source| Error::Spawn { program: "ffprobe".into(), source })?;
    parse_probe(&String::from_utf8_lossy(&output.stdout)).ok_or_else(|| Error::NoVideoStream(path.to_owned()))
}

/// Parse ffprobe's `key=value` output; `None` without a video stream.
fn parse_probe(text: &str) -> Option<VideoInfo> {
    let field = |key: &str| {
        text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('=')).map(str::trim).filter(|v| !v.is_empty() && *v != "N/A")
    };
    let number = |key: &str| field(key).and_then(|v| v.parse::<usize>().ok());
    let fps = field("avg_frame_rate")
        .and_then(|rate| {
            let (n, d) = rate.split_once('/')?;
            let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
            (d > 0.0 && n > 0.0).then(|| n / d)
        })
        .unwrap_or(FALLBACK_FPS);
    Some(VideoInfo {
        width: number("width")?,
        height: number("height")?,
        fps,
        frame_count: field("nb_frames").and_then(|n| n.parse().ok()),
        duration: field("duration").and_then(|d| d.parse().ok()).filter(|d: &f64| d.is_finite() && *d > 0.0),
    })
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

/// Video encoder choice (an ffmpeg encoder name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoder(pub String);

/// Encoders wajh offers, fastest first; hardware ones are only offered where the hardware exists.
const KNOWN_ENCODERS: [(&str, &str); 6] = [
    ("h264_nvenc", "NVENC H.264 · GPU"),
    ("hevc_nvenc", "NVENC HEVC · GPU"),
    ("h264_videotoolbox", "VideoToolbox H.264"),
    ("hevc_videotoolbox", "VideoToolbox HEVC"),
    ("libx264", "x264 H.264 · CPU"),
    ("libx265", "x265 HEVC · CPU"),
];
/// Used when ffmpeg lists none of the known encoders.
const DEFAULT_ENCODER: &str = "libx264";

impl Encoder {
    /// Fastest available: NVENC, `VideoToolbox`, else libx264. Blocking (runs ffmpeg).
    #[must_use]
    pub fn best_available() -> Self {
        Self::available().into_iter().next().unwrap_or_else(|| Self(DEFAULT_ENCODER.to_owned()))
    }

    /// Every known encoder this ffmpeg build and machine can use, fastest first. Blocking.
    #[must_use]
    pub fn available() -> Vec<Self> {
        let listed = ffmpeg()
            .arg("-encoders")
            .stderr(Stdio::null())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        let nvidia = std::path::Path::new("/dev/nvidia0").exists() || cfg!(target_os = "windows");
        usable_encoders(&listed, nvidia, cfg!(target_os = "macos"))
    }

    /// A readable name for known encoders, else the ffmpeg name.
    #[must_use]
    pub fn label(&self) -> &str {
        KNOWN_ENCODERS.iter().find(|(name, _)| *name == self.0).map_or(&self.0, |(_, label)| label)
    }

    fn quality_args(&self, cmd: &mut Command) {
        match self.0.as_str() {
            "h264_nvenc" | "hevc_nvenc" => cmd.args(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19"]),
            "libx264" | "libx265" => cmd.args(["-preset", "fast", "-crf", "18"]),
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

/// Known encoders present in an `ffmpeg -encoders` listing whose hardware is present.
fn usable_encoders(listing: &str, nvidia: bool, macos: bool) -> Vec<Encoder> {
    let listed: Vec<&str> = listing.lines().filter_map(|line| line.split_whitespace().nth(1)).collect();
    KNOWN_ENCODERS
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| listed.contains(name))
        .filter(|name| (nvidia || !name.ends_with("_nvenc")) && (macos || !name.ends_with("_videotoolbox")))
        .map(|name| Encoder(name.to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_probe_output() {
        let info = parse_probe("width=1920\nheight=1080\navg_frame_rate=30000/1001\nnb_frames=N/A\nduration=10.010000\n").unwrap();
        assert_eq!((info.width, info.height, info.frame_count), (1920, 1080, None));
        assert!((info.fps - 29.97).abs() < 0.01);
        assert_eq!(info.total_frames(), Some(300));
        assert!(parse_probe("duration=3.0\n").is_none());
    }

    #[test]
    fn falls_back_when_the_rate_is_unknown() {
        let info = parse_probe("width=64\nheight=48\navg_frame_rate=0/0\nnb_frames=20\n").unwrap();
        assert_eq!(info.fps, FALLBACK_FPS);
        assert_eq!(info.seconds(), Some(20.0 / FALLBACK_FPS));
    }

    #[test]
    fn filters_encoders_by_hardware() {
        let listing = " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)";
        let names = |nvidia, macos| usable_encoders(listing, nvidia, macos).into_iter().map(|e| e.0).collect::<Vec<_>>();
        assert_eq!(names(true, false), ["h264_nvenc", "libx264"]);
        assert_eq!(names(false, true), ["h264_videotoolbox", "libx264"]);
        assert_eq!(Encoder("libx264".into()).label(), "x264 H.264 · CPU");
    }
}
