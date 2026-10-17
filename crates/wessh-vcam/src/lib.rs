//! Virtual cameras: other apps (Zoom, Teams, Discord, browsers) see wessh as a webcam.
//!
//! - **Linux**: a v4l2loopback device, written directly as YUYV ([`loopback`]).
//! - **Windows 11**: a Media Foundation virtual camera, in two halves:
//!   - **The media source** (feature `source`, built as `wessh-vcam.dll`). Windows' camera frame
//!     server (and the Frame Server Monitor, and apps that open the camera) load it in-process
//!     by its CLSID, which must be registered under `HKLM` and point at a file Local Service can
//!     read, so [`install`] copies the DLL to `%ProgramFiles%\wessh` (administrator rights,
//!     once). It serves NV12 frames it reads from a shared file ([`layout`]), black while there
//!     are none.
//!   - **The client**, used by wessh: writes frames into that file and keeps the camera
//!     registered with `MFCreateVirtualCamera` while a [`VirtualCamera`] lives (session
//!     lifetime, current user only: no administrator rights and nothing left behind on exit).
//!     `MFCreateVirtualCamera` exists from Windows 11 (build 22000); it is looked up at run
//!     time, so wessh still starts on Windows 10 and [`status`] reports [`Status::Unsupported`].
//! - **macOS**: none yet (wessh streams to OBS instead).
//!
//! This is the only crate in the workspace with `unsafe`, and only in the Windows modules (COM
//! and Win32 cannot be called without it); every unsafe block says why it is sound.

pub mod convert;
pub mod layout;
pub mod loopback;

#[cfg(windows)]
#[allow(unsafe_code, reason = "Media Foundation, COM and Win32 calls")]
mod client;
#[cfg(windows)]
#[allow(unsafe_code, reason = "file mapping shared with the media source")]
mod shared;
#[cfg(all(windows, feature = "source"))]
#[allow(unsafe_code, reason = "the COM media source and its DLL exports")]
mod source;
#[cfg(not(windows))]
mod unsupported;

#[cfg(windows)]
pub use client::{install, is_elevated, run_elevated, status, uninstall};
pub use loopback::{DEFAULT_LOOPBACK, LOOPBACK_MODPROBE, LoopbackStatus};
#[cfg(not(windows))]
pub use unsupported::{install, is_elevated, run_elevated, status, uninstall};

/// Where a [`VirtualCamera`] sends frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target<'a> {
    /// A v4l2loopback device such as `/dev/video10` (Linux).
    Loopback(&'a str),
    /// The Windows 11 virtual camera.
    Windows,
}

/// An open virtual camera taking BGR frames of one size; other apps stop seeing it on drop.
#[derive(Debug)]
pub struct VirtualCamera {
    backend: Backend,
}

#[derive(Debug)]
enum Backend {
    #[cfg(target_os = "linux")]
    Loopback(loopback::Loopback),
    #[cfg(windows)]
    Windows(client::Camera),
}

impl Backend {
    fn open(target: Target<'_>, width: usize, height: usize, fps: u32) -> Result<Self> {
        match target {
            #[cfg(target_os = "linux")]
            Target::Loopback(path) => Ok(Self::Loopback(loopback::Loopback::open(path, width, height)?)),
            #[cfg(windows)]
            Target::Windows => Ok(Self::Windows(client::Camera::start(width as u32, height as u32, fps)?)),
            #[cfg(not(target_os = "linux"))]
            Target::Loopback(_) => Err(Error::LoopbackUnsupported),
            #[cfg(not(windows))]
            Target::Windows => {
                let _ = (width, height, fps);
                Err(Error::Unsupported)
            }
        }
    }
}

impl VirtualCamera {
    /// Open `target` for `width`×`height` frames at `fps`.
    pub fn open(target: Target<'_>, width: usize, height: usize, fps: u32) -> Result<Self> {
        Ok(Self { backend: Backend::open(target, width, height, fps)? })
    }

    /// Send one BGR frame of the size the camera was opened for.
    pub fn write(&mut self, bgr: &[u8]) -> Result<()> {
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            let _ = bgr;
            match self.backend {}
        }
        #[cfg(any(target_os = "linux", windows))]
        match &mut self.backend {
            #[cfg(target_os = "linux")]
            Backend::Loopback(device) => Ok(device.write(bgr)?),
            #[cfg(windows)]
            Backend::Windows(camera) => {
                camera.write(bgr);
                Ok(())
            }
        }
    }
}

/// The media source's COM class. Registered by [`install`], passed to `MFCreateVirtualCamera`.
pub const CLSID: u128 = 0x2a39cd43_5aba_4330_86a0_d4ae62be9b01;
/// `CLSID` in registry form.
pub const CLSID_STRING: &str = "{2A39CD43-5ABA-4330-86A0-D4AE62BE9B01}";
/// The camera's name in other apps' camera lists.
pub const FRIENDLY_NAME: &str = "wessh";
/// The media source's file name, beside `wessh.exe` in a release and in `%ProgramFiles%\wessh`
/// once installed.
pub const DLL_NAME: &str = "wessh-vcam.dll";
/// What `cargo rustc --crate-type cdylib` names the DLL (development builds).
pub const CARGO_DLL_NAME: &str = "wessh_vcam.dll";
/// Registry value (under the CLSID key) holding the installed source's [`layout::PROTOCOL`].
pub const PROTOCOL_VALUE: &str = "WesshProtocol";

/// Whether the Windows virtual camera can be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Status {
    /// Not Windows 11 (or not Windows): there is no virtual camera API.
    #[default]
    Unsupported,
    /// Windows 11, but the media source is not installed.
    NotInstalled,
    /// An older wessh installed a media source that speaks another protocol; reinstall it.
    Outdated,
    /// Ready to start.
    Ready,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the wessh virtual camera needs Windows 11")]
    Unsupported,
    #[error(
        "the wessh virtual camera is not installed (or is from another version); install it from the app or with `wessh camera install`"
    )]
    NotInstalled,
    #[error("{width}x{height} at {fps} fps cannot be sent to the virtual camera (even sizes up to 8192, 1-240 fps)")]
    Format { width: u32, height: u32, fps: u32 },
    #[error("the administrator prompt was declined")]
    Cancelled,
    #[error("v4l2loopback output is Linux-only; use a udp:// target with OBS")]
    LoopbackUnsupported,
    #[error("the v4l2loopback device took {got:?} instead of {expected:?}")]
    LoopbackSize { expected: (usize, usize), got: (usize, usize) },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[cfg(windows)]
    #[error(transparent)]
    Windows(#[from] windows_core::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `%ProgramData%\wessh\vcam-frames`: the shared frames file ([`layout`]). Readable by every
/// account (Local Service runs the frame server), unlike anything under the user's profile.
#[must_use]
pub fn frames_path() -> std::path::PathBuf {
    let data = std::env::var_os("ProgramData").map_or_else(|| std::path::PathBuf::from(r"C:\ProgramData"), std::path::PathBuf::from);
    data.join("wessh").join("vcam-frames")
}

/// `%ProgramFiles%\wessh`: where [`install`] puts the media source, writable only by
/// administrators (the frame server must never load a DLL a user could swap).
#[must_use]
pub fn install_dir() -> std::path::PathBuf {
    let programs = std::env::var_os("ProgramW6432")
        .or_else(|| std::env::var_os("ProgramFiles"))
        .map_or_else(|| std::path::PathBuf::from(r"C:\Program Files"), std::path::PathBuf::from);
    programs.join("wessh")
}

/// The media source DLL in `dir` (the folder of `wessh.exe`): the release name, or the name a
/// development build gives it.
#[must_use]
pub fn find_dll(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    [DLL_NAME, CARGO_DLL_NAME].into_iter().map(|name| dir.join(name)).find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    #[test]
    fn clsid_string_matches_the_number() {
        let hex = format!("{:032X}", super::CLSID);
        let expected = format!("{{{}-{}-{}-{}-{}}}", &hex[..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..]);
        assert_eq!(super::CLSID_STRING, expected);
    }
}
