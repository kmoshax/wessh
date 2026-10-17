//! v4l2loopback (Linux): frames are written to the device directly as YUYV, so live mode needs
//! no ffmpeg.

/// One device at `/dev/video10`, labelled wessh; exclusive caps let browsers discover it.
pub const LOOPBACK_OPTIONS: &str = "devices=1 video_nr=10 card_label=wessh exclusive_caps=1";
/// The device selected by `video_nr` in [`LOOPBACK_OPTIONS`].
pub const DEFAULT_LOOPBACK: &str = "/dev/video10";

/// The manual equivalent of loading the camera for the current boot.
#[must_use]
pub fn loopback_modprobe() -> String {
    format!("sudo modprobe v4l2loopback {LOOPBACK_OPTIONS}")
}

/// Package installation guidance for the current distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstallHint {
    pub command: Option<&'static str>,
    pub note: &'static str,
}

/// Read distribution IDs as data, without sourcing the file as a shell script.
#[must_use]
pub fn parse_os_release(content: &str) -> InstallHint {
    let mut id = "";
    let mut like = "";
    for line in content.lines() {
        let Some((key, value)) = line.trim().split_once('=') else { continue };
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        match key.trim() {
            "ID" => id = value,
            "ID_LIKE" => like = value,
            _ => {}
        }
    }
    std::iter::once(id).chain(like.split_whitespace()).find_map(distro_hint).unwrap_or(InstallHint {
        command: None,
        note: "Install v4l2loopback for your running kernel using your distribution's package manager, then enable the camera.",
    })
}

fn distro_hint(id: &str) -> Option<InstallHint> {
    let command = match id {
        "arch" | "cachyos" | "manjaro" | "endeavouros" => "sudo pacman -S v4l2loopback-dkms",
        "debian" | "ubuntu" | "linuxmint" | "pop" => "sudo apt install v4l2loopback-dkms",
        "fedora" => {
            return Some(InstallHint {
                command: Some("sudo dnf install akmod-v4l2loopback"),
                note: "Requires RPM Fusion. Install the driver in a terminal, then enable the camera.",
            });
        }
        "opensuse" | "opensuse-leap" | "opensuse-tumbleweed" | "suse" => "sudo zypper install v4l2loopback-kmp-default",
        _ => return None,
    };
    Some(InstallHint { command: Some(command), note: "Install the driver in a terminal, then enable the camera." })
}

/// Distribution-specific instructions; an unreadable release file gives a generic hint.
#[must_use]
pub fn install_hint() -> InstallHint {
    parse_os_release(&std::fs::read_to_string("/etc/os-release").unwrap_or_default())
}

/// Check the running kernel's module index without requiring administrator rights.
#[cfg(target_os = "linux")]
#[must_use]
pub fn is_installed() -> bool {
    std::process::Command::new("modinfo").arg("v4l2loopback").output().is_ok_and(|output| output.status.success())
}

/// How to request administrator rights for camera setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnablePrompt {
    Graphical,
    Terminal,
}

/// Closing the password dialog is a normal outcome, separate from setup errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnableOutcome {
    Enabled,
    Cancelled,
}

#[derive(Debug, thiserror::Error)]
pub enum EnableError {
    #[error("pkexec is missing; install polkit to enable the camera from the app")]
    PkexecMissing,
    #[error("v4l2loopback is not installed for the running kernel; install the driver, then try again")]
    ModuleNotInstalled,
    #[error("could not start {program}: {source}")]
    Launch { program: &'static str, source: std::io::Error },
    #[error("could not finish virtual camera setup: {stderr}")]
    Failed { stderr: String },
}

/// Load now and persist for future boots in one elevated shell, so only one prompt is needed.
#[cfg(target_os = "linux")]
pub fn enable(prompt: EnablePrompt) -> Result<EnableOutcome, EnableError> {
    use std::process::{Command, Stdio};
    let program = match prompt {
        EnablePrompt::Graphical => "pkexec",
        EnablePrompt::Terminal => "sudo",
    };
    let mut command = Command::new(program);
    if prompt == EnablePrompt::Graphical {
        // A GUI without a polkit agent should report an error, not wait for a hidden terminal prompt.
        command.arg("--disable-internal-agent");
    }
    let output = command
        .args(["/bin/sh", "-c", &enable_script()])
        .stdin(Stdio::inherit())
        .output()
        .map_err(|source| launch_error(program, source))?;
    enable_result(prompt, output.status.code(), &String::from_utf8_lossy(&output.stderr))
}

#[cfg(target_os = "linux")]
fn launch_error(program: &'static str, source: std::io::Error) -> EnableError {
    if program == "pkexec" && source.kind() == std::io::ErrorKind::NotFound {
        EnableError::PkexecMissing
    } else {
        EnableError::Launch { program, source }
    }
}

#[cfg(target_os = "linux")]
fn enable_script() -> String {
    /// systemd modules-load and modprobe configuration owned by this application.
    const MODULES_FILE: &str = "/etc/modules-load.d/wessh-v4l2loopback.conf";
    const OPTIONS_FILE: &str = "/etc/modprobe.d/wessh-v4l2loopback.conf";
    /// Root writes the configuration; everyone can read it (files 0644, directories 0755).
    const CONFIG_UMASK: &str = "022";
    format!(
        "set -eu\n\
         export PATH=/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C\n\
         umask {CONFIG_UMASK}\n\
         modprobe v4l2loopback {LOOPBACK_OPTIONS}\n\
         mkdir -p /etc/modules-load.d /etc/modprobe.d\n\
         printf '%s\\n' 'v4l2loopback' > {MODULES_FILE}\n\
         printf '%s\\n' 'options v4l2loopback {LOOPBACK_OPTIONS}' > {OPTIONS_FILE}\n"
    )
}

#[cfg(target_os = "linux")]
fn enable_result(prompt: EnablePrompt, code: Option<i32>, stderr: &str) -> Result<EnableOutcome, EnableError> {
    /// pkexec's documented status when the user dismisses the authentication dialog.
    const PKEXEC_CANCELLED: i32 = 126;
    if code == Some(0) {
        return Ok(EnableOutcome::Enabled);
    }
    if prompt == EnablePrompt::Graphical && code == Some(PKEXEC_CANCELLED) {
        return Ok(EnableOutcome::Cancelled);
    }
    if stderr.contains("modprobe:") && stderr.contains("Module v4l2loopback not found") {
        return Err(EnableError::ModuleNotInstalled);
    }
    let stderr = if stderr.trim().is_empty() {
        code.map_or_else(|| "process terminated by a signal".to_owned(), |code| format!("process exited with status {code}"))
    } else {
        stderr.trim().to_owned()
    };
    Err(EnableError::Failed { stderr })
}

/// Whether frames can be sent to a v4l2loopback virtual camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopbackStatus {
    /// Not Linux: virtual-camera output goes elsewhere (Windows 11) or over a network stream.
    #[default]
    Unsupported,
    /// Linux, but the module is not installed for the running kernel.
    NotInstalled,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distro_ids_select_the_install_command() {
        for (ids, expected) in [
            (&["arch", "cachyos", "manjaro", "endeavouros"][..], "sudo pacman -S v4l2loopback-dkms"),
            (&["debian", "ubuntu", "linuxmint", "pop"][..], "sudo apt install v4l2loopback-dkms"),
            (&["fedora"][..], "sudo dnf install akmod-v4l2loopback"),
            (&["opensuse", "opensuse-leap", "opensuse-tumbleweed", "suse"][..], "sudo zypper install v4l2loopback-kmp-default"),
        ] {
            for id in ids {
                assert_eq!(parse_os_release(&format!("ID={id}\n")).command, Some(expected));
            }
        }
        assert!(parse_os_release("ID=fedora").note.contains("RPM Fusion"));
    }

    #[test]
    fn release_parsing_handles_quotes_and_prefers_id_over_ancestry() {
        let arch = parse_os_release("ID=arch");
        let debian = parse_os_release("ID=debian");
        assert_eq!(parse_os_release("ID_LIKE=debian\nID=\"arch\"\n"), arch);
        assert_eq!(parse_os_release("# ID=fedora\nNAME=Example\nID='custom'\nID_LIKE=\"unknown arch debian\""), arch);
        assert_eq!(parse_os_release("\n ID = 'linuxmint' \nID_LIKE=ubuntu"), debian);
        assert_eq!(parse_os_release("ID=custom\nID_LIKE='ubuntu debian'"), debian);
        assert_eq!(parse_os_release("ID=archlinux\nID_LIKE=notarch").command, None);
        assert_eq!(parse_os_release("bad line\nID\nID_LIKE=\n"), parse_os_release(""));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_loads_before_persisting_and_shares_options() {
        let script = enable_script();
        let load = loopback_modprobe().trim_start_matches("sudo ").to_owned();
        assert!(script.contains(&load));
        assert!(script.find(&load).unwrap() < script.find("mkdir").unwrap());
        assert!(script.starts_with("set -eu\n"));
        assert!(script.contains("LC_ALL=C"));
        assert!(script.contains("printf '%s\\n' 'v4l2loopback' > /etc/modules-load.d/wessh-v4l2loopback.conf"));
        assert!(
            script.contains(&format!("printf '%s\\n' 'options v4l2loopback {LOOPBACK_OPTIONS}' > /etc/modprobe.d/wessh-v4l2loopback.conf"))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn setup_distinguishes_cancellation_missing_module_and_other_failures() {
        let graphical = EnablePrompt::Graphical;
        assert_eq!(enable_result(graphical, Some(0), "").unwrap(), EnableOutcome::Enabled);
        assert_eq!(enable_result(graphical, Some(126), "dismissed").unwrap(), EnableOutcome::Cancelled);
        assert!(matches!(
            enable_result(graphical, Some(1), "modprobe: FATAL: Module v4l2loopback not found in directory /lib/modules/test"),
            Err(EnableError::ModuleNotInstalled)
        ));
        for (prompt, code, stderr) in [
            (graphical, Some(127), "Not authorized"),
            (graphical, Some(1), "modprobe: ERROR: could not insert 'v4l2loopback': Key was rejected by service"),
            (graphical, Some(1), "cannot create /etc/modprobe.d/wessh-v4l2loopback.conf: Read-only file system"),
            (EnablePrompt::Terminal, Some(126), "sudo failed"),
        ] {
            let Err(EnableError::Failed { stderr: actual }) = enable_result(prompt, code, stderr) else { panic!("expected failure") };
            assert_eq!(actual, stderr);
        }
        assert!(enable_result(graphical, None, "").unwrap_err().to_string().contains("signal"));
        assert!(enable_result(graphical, Some(1), "").unwrap_err().to_string().contains("status 1"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn missing_pkexec_is_distinct_from_other_launch_errors() {
        use std::io::{Error, ErrorKind};
        assert!(matches!(launch_error("pkexec", Error::from(ErrorKind::NotFound)), EnableError::PkexecMissing));
        assert!(matches!(launch_error("sudo", Error::from(ErrorKind::NotFound)), EnableError::Launch { .. }));
        assert!(matches!(launch_error("pkexec", Error::from(ErrorKind::PermissionDenied)), EnableError::Launch { .. }));
    }
}
