use crate::ffmpeg::{ffmpeg, raw_bgr_input, spawn};
use crate::{Error, Result};
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use wajh_core::Frame;

/// The v4l2loopback device created by [`crate::LOOPBACK_MODPROBE`] (`video_nr=10`).
pub const DEFAULT_LOOPBACK: &str = "/dev/video10";
/// A local UDP MPEG-TS stream that OBS can add as a media source.
pub const DEFAULT_STREAM: &str = "udp://127.0.0.1:27000";

/// Where live output goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputTarget {
    /// A v4l2loopback device such as `/dev/video10` (Linux).
    V4l2(String),
    /// A network stream (`udp://`, `tcp://`, `srt://`) that OBS can add as a media source.
    Stream(String),
}

impl OutputTarget {
    /// Classify a user-supplied target string.
    #[must_use]
    pub fn parse(target: &str) -> Self {
        if ["udp://", "tcp://", "srt://"].iter().any(|p| target.starts_with(p)) {
            Self::Stream(target.to_owned())
        } else {
            Self::V4l2(target.to_owned())
        }
    }

    #[must_use]
    pub fn default_for_platform() -> Self {
        if cfg!(target_os = "linux") { Self::V4l2(DEFAULT_LOOPBACK.into()) } else { Self::Stream(DEFAULT_STREAM.into()) }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::V4l2(s) | Self::Stream(s) => s,
        }
    }
}

/// An ffmpeg process consuming raw BGR frames on stdin.
#[derive(Debug)]
pub struct FrameSink {
    process: Child,
    stdin: Option<ChildStdin>,
    size: (usize, usize),
}

impl FrameSink {
    /// Live output to a virtual camera or network stream.
    pub fn live(target: &OutputTarget, width: usize, height: usize, fps: u32) -> Result<Self> {
        let mut cmd = ffmpeg();
        raw_bgr_input(&mut cmd, width, height, f64::from(fps));
        match target {
            OutputTarget::Stream(url) => cmd.args(["-c:v", "mpeg1video", "-q:v", "3", "-bf", "0", "-f", "mpegts", url]),
            OutputTarget::V4l2(device) if cfg!(target_os = "linux") => cmd.args(["-f", "v4l2", "-pix_fmt", "yuyv422", device]),
            OutputTarget::V4l2(_) => return Err(Error::Unsupported("v4l2 output is Linux-only; use a udp:// target with OBS")),
        };
        Self::spawn(cmd, width, height)
    }

    pub(crate) fn spawn(mut cmd: Command, width: usize, height: usize) -> Result<Self> {
        let mut process = spawn(cmd.stdin(Stdio::piped()))?;
        let stdin = process.stdin.take();
        Ok(Self { process, stdin, size: (width, height) })
    }

    pub fn write(&mut self, frame: &Frame) -> Result<()> {
        let got = (frame.width, frame.height);
        if got != self.size {
            return Err(Error::FrameSize { expected: self.size, got });
        }
        let stdin = self.stdin.as_mut().ok_or(Error::Unsupported("output already closed"))?;
        stdin.write_all(&frame.data)?;
        Ok(())
    }

    /// Close the stream and wait for ffmpeg to finish writing.
    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.process.wait()?;
        if status.success() { Ok(()) } else { Err(Error::Exited(status)) }
    }
}

impl Drop for FrameSink {
    fn drop(&mut self) {
        if self.stdin.take().is_some() {
            let _ = self.process.kill();
        }
        let _ = self.process.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::OutputTarget;

    #[test]
    fn parses_targets() {
        assert_eq!(OutputTarget::parse("udp://127.0.0.1:1"), OutputTarget::Stream("udp://127.0.0.1:1".into()));
        assert_eq!(OutputTarget::parse("/dev/video10"), OutputTarget::V4l2("/dev/video10".into()));
    }
}
