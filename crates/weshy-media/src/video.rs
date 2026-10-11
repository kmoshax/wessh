use crate::ffmpeg::{ffmpeg, ffmpeg_program, raw_bgr_input, spawn};
use crate::{Error, FrameSink, Result};
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};
use weshy_core::Frame;

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

/// Read the first video stream's geometry and frame rate from what `ffmpeg -i` prints, so no
/// separate ffprobe is needed.
pub fn probe(path: &str) -> Result<VideoInfo> {
    let output = ffmpeg_program()
        .args(["-hide_banner", "-nostdin", "-i", path])
        .output()
        .map_err(|source| Error::Spawn { program: "ffmpeg".into(), source })?;
    parse_probe(&String::from_utf8_lossy(&output.stderr)).ok_or_else(|| Error::NoVideoStream(path.to_owned()))
}

/// Parse `ffmpeg -i`'s input description; `None` without a video stream.
///
/// ```text
///   Duration: 00:00:10.01, start: 0.000000, bitrate: 484 kb/s
///   Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv), 1920x1080 [SAR 1:1 DAR 16:9], 29.97 fps, ...
/// ```
fn parse_probe(text: &str) -> Option<VideoInfo> {
    let video = text.lines().find(|line| line.trim_start().starts_with("Stream #") && line.contains(": Video: "))?;
    let parts: Vec<&str> = video.split(", ").map(str::trim).collect();
    let (width, height) = parts.iter().find_map(|part| {
        let (w, h) = part.split_whitespace().next()?.split_once('x')?;
        let (w, h) = (w.parse::<usize>().ok()?, h.parse::<usize>().ok()?);
        (w > 0 && h > 0).then_some((w, h))
    })?;
    let fps = parts
        .iter()
        .find_map(|part| part.strip_suffix(" fps")?.parse::<f64>().ok())
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .unwrap_or(FALLBACK_FPS);
    let duration = text.lines().find_map(|line| {
        let clock = line.trim_start().strip_prefix("Duration: ")?.split(',').next()?;
        let mut fields = clock.split(':').map(|f| f.parse::<f64>().ok());
        let (h, m, s) = (fields.next()??, fields.next()??, fields.next()??);
        Some(h.mul_add(3600.0, m.mul_add(60.0, s)))
    });
    Some(VideoInfo { width, height, fps, frame_count: None, duration: duration.filter(|d| d.is_finite() && *d > 0.0) })
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

/// Encoders weshy offers, fastest first; hardware ones are only offered where the hardware exists.
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
        let text = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Duration: 00:00:10.01, start: 0.000000, bitrate: 484 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 482 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
At least one output file must be specified";
        let info = parse_probe(text).unwrap();
        assert_eq!((info.width, info.height, info.frame_count), (1920, 1080, None));
        assert!((info.fps - 29.97).abs() < 0.01);
        assert_eq!(info.total_frames(), Some(300));
        assert!(parse_probe("  Duration: 00:00:03.00, start: 0.0\n  Stream #0:0: Audio: aac, 44100 Hz").is_none());
    }

    #[test]
    fn falls_back_when_the_rate_or_duration_is_unknown() {
        let info = parse_probe("  Duration: N/A, bitrate: N/A\n  Stream #0:0: Video: png, rgb24(pc), 64x48, 25 tbr").unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        assert_eq!(info.fps, FALLBACK_FPS);
        assert_eq!(info.duration, None);
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
