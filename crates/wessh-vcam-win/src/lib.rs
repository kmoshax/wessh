//! The Windows 11 virtual camera: other apps (Zoom, Teams, Discord, browsers) see wessh as a
//! webcam, without OBS. This is the only crate in the workspace that may use `unsafe`: COM and
//! Win32 cannot be called without it, and every unsafe block says why it is sound.
//!
//! Two halves, one crate:
//!
//! - **The media source** (feature `source`, built as `wessh-vcam.dll`). Windows' camera frame
//!   server (and the Frame Server Monitor, and apps that open the camera) load it in-process by
//!   its CLSID, which must be registered under `HKLM` and point at a file Local Service can read,
//!   so [`install`] copies the DLL to `%ProgramFiles%\wessh` (needs administrator rights, once).
//!   It serves NV12 frames it reads from a shared file ([`layout`]), black while there are none.
//! - **The client** ([`VirtualCamera`]), used by wessh: writes frames into that file and keeps
//!   the camera registered with `MFCreateVirtualCamera` for as long as it lives (session
//!   lifetime, current user only: no administrator rights and nothing left behind on exit).
//!
//! `MFCreateVirtualCamera` exists from Windows 11 (build 22000); it is looked up at run time, so
//! wessh still starts on Windows 10 and [`status`] reports [`Status::Unsupported`].

pub mod layout;

#[cfg(windows)]
mod client;
#[cfg(windows)]
mod shared;
#[cfg(all(windows, feature = "source"))]
mod source;
#[cfg(not(windows))]
mod unsupported;

#[cfg(windows)]
pub use client::{VirtualCamera, install, is_elevated, run_elevated, status, uninstall};
#[cfg(not(windows))]
pub use unsupported::{install, is_elevated, run_elevated, status, uninstall};

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
pub const CARGO_DLL_NAME: &str = "wessh_vcam_win.dll";
/// Registry value (under the CLSID key) holding the installed source's [`layout::PROTOCOL`].
pub const PROTOCOL_VALUE: &str = "WesshProtocol";

/// Whether the virtual camera can be used.
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
