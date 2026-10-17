use crate::ffmpeg::{ffmpeg, raw_bgr_input, spawn};
use crate::{Error, Result};
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use wessh_core::Frame;

pub use wessh_vcam::DEFAULT_LOOPBACK;
/// A local UDP MPEG-TS stream that OBS can add as a media source.
pub const DEFAULT_STREAM: &str = "udp://127.0.0.1:27000";
/// The target string of the built-in Windows 11 virtual camera.
pub const WINDOWS_CAMERA: &str = "windows-camera";

/// Where live output goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputTarget {
    /// A v4l2loopback device such as `/dev/video10` (Linux).
    V4l2(String),
    /// A network stream (`udp://`, `tcp://`, `srt://`) that OBS can add as a media source.
    Stream(String),
    /// The built-in virtual camera (Windows 11, `wessh-vcam`).
    WindowsCamera,
}

impl OutputTarget {
    /// Whether writing to this target goes through ffmpeg.
    #[must_use]
    pub const fn needs_ffmpeg(&self) -> bool {
        matches!(self, Self::Stream(_))
    }

    /// Classify a user-supplied target string.
    #[must_use]
    pub fn parse(target: &str) -> Self {
        if target == WINDOWS_CAMERA {
            Self::WindowsCamera
        } else if ["udp://", "tcp://", "srt://"].iter().any(|p| target.starts_with(p)) {
            Self::Stream(target.to_owned())
        } else {
            Self::V4l2(target.to_owned())
        }
    }

    /// v4l2loopback on Linux, the built-in camera on Windows 11 (installed or not: the app offers
    /// to install it), an OBS stream elsewhere.
    #[must_use]
    pub fn default_for_platform() -> Self {
        if cfg!(target_os = "linux") {
            Self::V4l2(DEFAULT_LOOPBACK.into())
        } else if wessh_vcam::status() != crate::WindowsCameraStatus::Unsupported {
            Self::WindowsCamera
        } else {
            Self::Stream(DEFAULT_STREAM.into())
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::V4l2(s) | Self::Stream(s) => s,
            Self::WindowsCamera => WINDOWS_CAMERA,
        }
    }
}

/// Where a [`FrameSink`]'s frames go.
enum Backend {
    /// An ffmpeg process reading raw BGR on stdin (streams, video files).
    Ffmpeg { process: Child, stdin: Option<ChildStdin> },
    /// A virtual camera (v4l2loopback, Windows 11), written directly: live mode needs no ffmpeg.
    Camera(wessh_vcam::VirtualCamera),
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ffmpeg { process, .. } => f.debug_struct("Ffmpeg").field("pid", &process.id()).finish(),
            Self::Camera(camera) => camera.fmt(f),
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
            OutputTarget::V4l2(path) => Self::camera(wessh_vcam::Target::Loopback(path), width, height, fps),
            OutputTarget::WindowsCamera => Self::camera(wessh_vcam::Target::Windows, width, height, fps),
        }
    }

    fn camera(target: wessh_vcam::Target<'_>, width: usize, height: usize, fps: u32) -> Result<Self> {
        let camera = wessh_vcam::VirtualCamera::open(target, width, height, fps)?;
        Ok(Self { backend: Backend::Camera(camera), size: (width, height) })
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
            Backend::Camera(camera) => camera.write(&frame.data)?,
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
            Backend::Camera(_) => Ok(()),
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
            Backend::Camera(_) => {}
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
        assert_eq!(OutputTarget::parse(super::WINDOWS_CAMERA), OutputTarget::WindowsCamera);
        assert_eq!(OutputTarget::WindowsCamera.as_str(), super::WINDOWS_CAMERA);
    }
}
