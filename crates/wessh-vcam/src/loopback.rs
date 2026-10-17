//! v4l2loopback (Linux): frames are written to the device directly as YUYV, so live mode needs
//! no ffmpeg.

/// The command that creates a v4l2loopback device wessh can write to (`/dev/video10`, named "wessh").
pub const LOOPBACK_MODPROBE: &str = "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=wessh exclusive_caps=1";
/// The device [`LOOPBACK_MODPROBE`] creates (`video_nr=10`).
pub const DEFAULT_LOOPBACK: &str = "/dev/video10";

/// Whether frames can be sent to a v4l2loopback virtual camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopbackStatus {
    /// Not Linux: virtual-camera output goes elsewhere (Windows 11) or over a network stream.
    #[default]
    Unsupported,
    /// Linux, but the `v4l2loopback` kernel module is not loaded.
    NotLoaded,
    /// The module is loaded.
    Loaded,
}

/// An open v4l2loopback device taking YUYV frames of one size.
#[cfg(target_os = "linux")]
pub(crate) struct Loopback {
    device: v4l::Device,
    yuyv: Vec<u8>,
    width: usize,
    height: usize,
}

#[cfg(target_os = "linux")]
impl std::fmt::Debug for Loopback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Loopback")
    }
}

#[cfg(target_os = "linux")]
impl Loopback {
    /// Open `path` for `width`×`height` YUYV output.
    pub(crate) fn open(path: &str, width: usize, height: usize) -> crate::Result<Self> {
        use v4l::video::Output;
        /// Bytes per pixel of packed YUYV.
        const YUYV_BYTES: usize = 2;
        let device = v4l::Device::with_path(path)?;
        let format = v4l::Format::new(width as u32, height as u32, v4l::FourCC::new(b"YUYV"));
        let set = device.set_format(&format)?;
        let got = (set.width as usize, set.height as usize);
        if got != (width, height) {
            return Err(crate::Error::LoopbackSize { expected: (width, height), got });
        }
        Ok(Self { device, yuyv: vec![0; width * height * YUYV_BYTES], width, height })
    }

    pub(crate) fn write(&mut self, bgr: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        crate::convert::bgr_to_yuyv(bgr, self.width, self.height, &mut self.yuyv);
        self.device.write_all(&self.yuyv)
    }
}
