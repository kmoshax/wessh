//! `wessh camera`: install, remove or check the built-in Windows 11 virtual camera.
//!
//! Installing copies `wessh-vcam.dll` (shipped beside `wessh.exe`) to `%ProgramFiles%\wessh` and
//! registers it for the camera frame server, which needs administrator rights: run without
//! them, the command starts itself again elevated (one UAC prompt) and waits. The app's Output
//! section calls [`install`] the same way.

use crate::cli::CameraAction;
use anyhow::{Context, Result, bail, ensure};
use wessh_media::{WindowsCameraStatus, windows_camera};

/// `wessh camera status` exit codes, which the Windows installer checks before installing or
/// removing the camera.
pub const EXIT_READY: i32 = 0;
pub const EXIT_UNSUPPORTED: i32 = 2;
pub const EXIT_NOT_INSTALLED: i32 = 3;
pub const EXIT_OUTDATED: i32 = 4;

pub fn run(action: CameraAction) -> Result<()> {
    match action {
        CameraAction::Install => install(),
        CameraAction::Uninstall => uninstall(),
        CameraAction::Status => {
            let status = windows_camera::status();
            println!("{}", describe(status));
            std::process::exit(match status {
                WindowsCameraStatus::Ready => EXIT_READY,
                WindowsCameraStatus::Unsupported => EXIT_UNSUPPORTED,
                WindowsCameraStatus::NotInstalled => EXIT_NOT_INSTALLED,
                WindowsCameraStatus::Outdated => EXIT_OUTDATED,
            })
        }
    }
}

/// One line for `status` and `doctor`.
pub fn describe(status: WindowsCameraStatus) -> &'static str {
    match status {
        WindowsCameraStatus::Unsupported => "not available (needs Windows 11; stream to OBS instead)",
        WindowsCameraStatus::NotInstalled => "not installed (`wessh camera install`)",
        WindowsCameraStatus::Outdated => "installed by another wessh version (`wessh camera install` updates it)",
        WindowsCameraStatus::Ready => "installed",
    }
}

/// Install (or update) the virtual camera, asking for administrator rights if needed.
pub fn install() -> Result<()> {
    if windows_camera::status() == WindowsCameraStatus::Unsupported {
        bail!("the virtual camera needs Windows 11; on Windows 10 stream to OBS instead");
    }
    let exe = std::env::current_exe().context("cannot locate the wessh executable")?;
    if windows_camera::is_elevated() {
        let folder = exe.parent().context("the executable has no folder")?;
        let dll = windows_camera::find_dll(folder)
            .with_context(|| format!("{} is missing beside wessh; reinstall wessh", windows_camera::DLL_NAME))?;
        let installed = windows_camera::install(&dll).context("could not install the virtual camera")?;
        println!("installed the virtual camera ({})", installed.display());
        return Ok(());
    }
    elevate(&exe, "camera install")?;
    ensure!(windows_camera::status() == WindowsCameraStatus::Ready, "the virtual camera was not installed");
    println!("installed the virtual camera");
    Ok(())
}

/// Remove the virtual camera, asking for administrator rights if needed.
pub fn uninstall() -> Result<()> {
    if windows_camera::is_elevated() {
        windows_camera::uninstall().context("could not remove the virtual camera")?;
    } else {
        let exe = std::env::current_exe().context("cannot locate the wessh executable")?;
        elevate(&exe, "camera uninstall")?;
    }
    println!("removed the virtual camera");
    Ok(())
}

/// Run `exe arguments` as administrator and wait for it to succeed.
fn elevate(exe: &std::path::Path, arguments: &str) -> Result<()> {
    let code = windows_camera::run_elevated(exe, arguments).context("could not run wessh as administrator")?;
    ensure!(code == 0, "`wessh {arguments}` as administrator failed (exit code {code})");
    Ok(())
}
