use crate::ffmpeg::{ffmpeg, raw_bgr_input, spawn};
use crate::{Error, Result};
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use wessh_core::Frame;

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

/// Where a [`FrameSink`]'s frames go.
enum Backend {
    /// An ffmpeg process reading raw BGR on stdin (streams, video files).
    Ffmpeg { process: Child, stdin: Option<ChildStdin> },
    /// A v4l2loopback device written directly as YUYV, so live mode needs no ffmpeg.
    #[cfg(target_os = "linux")]
    Loopback { device: v4l::Device, yuyv: Vec<u8> },
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ffmpeg { process, .. } => f.debug_struct("Ffmpeg").field("pid", &process.id()).finish(),
            #[cfg(target_os = "linux")]
            Self::Loopback { .. } => f.write_str("Loopback"),
        }
    }
}

/// Consumes BGR frames of one fixed size.
#[derive(Debug)]
pub struct FrameSink {
    backend: Backend,
    size: (usize, usize),
}

impl FrameSink {
    /// Live output to a virtual camera or network stream.
    pub fn live(target: &OutputTarget, width: usize, height: usize, fps: u32) -> Result<Self> {
        match target {
            OutputTarget::Stream(url) => {
                let mut cmd = ffmpeg();
                raw_bgr_input(&mut cmd, width, height, f64::from(fps));
                cmd.args(["-c:v", "mpeg1video", "-q:v", "3", "-bf", "0", "-f", "mpegts", url]);
                Self::spawn(cmd, width, height)
            }
            #[cfg(target_os = "linux")]
            OutputTarget::V4l2(path) => Self::loopback(path, width, height),
            #[cfg(not(target_os = "linux"))]
            OutputTarget::V4l2(_) => Err(Error::Unsupported("v4l2 output is Linux-only; use a udp:// target with OBS")),
        }
    }

    /// Open a v4l2loopback device for `width`×`height` YUYV output.
    #[cfg(target_os = "linux")]
    fn loopback(path: &str, width: usize, height: usize) -> Result<Self> {
        use v4l::video::Output;
        let device = v4l::Device::with_path(path)?;
        let format = v4l::Format::new(width as u32, height as u32, v4l::FourCC::new(b"YUYV"));
        let set = device.set_format(&format)?;
        if (set.width as usize, set.height as usize) != (width, height) {
            return Err(Error::FrameSize { expected: (width, height), got: (set.width as usize, set.height as usize) });
        }
        Ok(Self { backend: Backend::Loopback { device, yuyv: vec![0; width * height * 2] }, size: (width, height) })
    }

    pub(crate) fn spawn(mut cmd: Command, width: usize, height: usize) -> Result<Self> {
        let mut process = spawn(cmd.stdin(Stdio::piped()))?;
        let stdin = process.stdin.take();
        Ok(Self { backend: Backend::Ffmpeg { process, stdin }, size: (width, height) })
    }

    pub fn write(&mut self, frame: &Frame) -> Result<()> {
        let got = (frame.width, frame.height);
        if got != self.size {
            return Err(Error::FrameSize { expected: self.size, got });
        }
        match &mut self.backend {
            Backend::Ffmpeg { stdin, .. } => stdin.as_mut().ok_or(Error::Unsupported("output already closed"))?.write_all(&frame.data)?,
            #[cfg(target_os = "linux")]
            Backend::Loopback { device, yuyv } => {
                crate::convert::bgr_to_yuyv(&frame.data, frame.width, frame.height, yuyv);
                device.write_all(yuyv)?;
            }
        }
        Ok(())
    }

    /// Close the output and, for ffmpeg, wait for it to finish writing.
    pub fn finish(mut self) -> Result<()> {
        match &mut self.backend {
            Backend::Ffmpeg { process, stdin } => {
                drop(stdin.take());
                let status = process.wait()?;
                if status.success() { Ok(()) } else { Err(Error::Exited(status)) }
            }
            #[cfg(target_os = "linux")]
            Backend::Loopback { .. } => Ok(()),
        }
    }
}

impl Drop for FrameSink {
    fn drop(&mut self) {
        match &mut self.backend {
            Backend::Ffmpeg { process, stdin } => {
                if stdin.take().is_some() {
                    let _ = process.kill();
                }
                let _ = process.wait();
            }
            #[cfg(target_os = "linux")]
            Backend::Loopback { .. } => {}
        }
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
