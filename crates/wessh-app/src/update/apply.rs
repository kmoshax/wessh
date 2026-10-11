use super::{InstallKind, Update, binary_from_archive};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use wessh_core::process::background_command;

const UNIX_HELPER: &str = include_str!("../../../../scripts/update-unix.sh");
const WINDOWS_HELPER: &str = include_str!("../../../../scripts/update-windows.ps1");

#[derive(Clone, Debug)]
pub struct Prepared {
    update: Update,
    source: PathBuf,
    workspace: PathBuf,
}

fn unique_suffix() -> Result<String> {
    Ok(format!("{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()))
}

pub(super) fn stage(update: &Update, archive: &Path, folder: &Path) -> Result<Prepared> {
    let workspace = folder.join(format!("stage-{}", unique_suffix()?));
    fs::create_dir(&workspace)?;
    let source = match update.target.kind {
        InstallKind::WindowsInstaller | InstallKind::AppImage => archive.to_path_buf(),
        InstallKind::WindowsPortable => {
            let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
            for name in ["wessh.exe", "wessh-cli.exe"] {
                let mut entry = zip.by_name(name).with_context(|| format!("update archive is missing {name}"))?;
                ensure!(entry.is_file(), "invalid executable in update archive");
                std::io::copy(&mut entry, &mut File::create(workspace.join(name))?)?;
            }
            workspace.clone()
        }
        InstallKind::Binary => {
            let binary = workspace.join("wessh");
            binary_from_archive(archive, &binary)?;
            executable(&binary)?;
            binary
        }
        InstallKind::MacBundle => stage_bundle(archive, &workspace)?,
    };
    Ok(Prepared { update: update.clone(), source, workspace })
}

fn stage_bundle(archive: &Path, workspace: &Path) -> Result<PathBuf> {
    let mount = workspace.join("mount");
    fs::create_dir(&mount)?;
    successful(background_command("hdiutil").args(["attach", "-nobrowse", "-readonly", "-mountpoint"]).arg(&mount).arg(archive))?;
    let bundle = workspace.join("wessh.app");
    let copy = successful(background_command("ditto").arg(mount.join("wessh.app")).arg(&bundle))
        .and_then(|()| successful(background_command("codesign").args(["--verify", "--deep", "--strict"]).arg(&bundle)));
    let detach = successful(background_command("hdiutil").arg("detach").arg(&mount));
    copy?;
    detach?;
    Ok(bundle)
}

fn successful(command: &mut Command) -> Result<()> {
    let output = command.output().with_context(|| format!("could not run {}", command.get_program().to_string_lossy()))?;
    ensure!(
        output.status.success(),
        "{} failed: {}",
        command.get_program().to_string_lossy(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

#[cfg(unix)]
fn executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn executable(path: &Path) -> Result<()> {
    fs::metadata(path)?;
    Ok(())
}

/// Start a helper that waits for this process to exit before touching installed files.
/// The caller closes the app only after this succeeds. Failures restore and reopen the old app.
pub fn launch(prepared: &Prepared, restart: bool) -> Result<()> {
    let target = &prepared.update.target;
    let log = File::create(prepared.workspace.join("update.log"))?;
    let mut command = if matches!(target.kind, InstallKind::WindowsInstaller | InstallKind::WindowsPortable) {
        let probe = target.path.join(format!(".wessh-update-{}.tmp", unique_suffix()?));
        File::options().write(true).create_new(true).open(&probe).context("cannot write to the installation folder")?;
        fs::remove_file(probe)?;
        let script = prepared.workspace.join("apply.ps1");
        fs::write(&script, WINDOWS_HELPER)?;
        let config = WindowsConfig {
            pid: std::process::id(),
            installer: target.kind == InstallKind::WindowsInstaller,
            source: &prepared.source,
            target: &target.path,
            restart,
            arguments: std::env::args().skip(1).collect(),
        };
        let config_path = prepared.workspace.join("config.json");
        fs::write(&config_path, serde_json::to_vec(&config)?)?;
        let mut command = background_command("powershell.exe");
        command.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(script).arg(config_path);
        command
    } else {
        let parent = target.path.parent().context("installation has no parent directory")?;
        let stem = target.path.file_name().context("installation has no file name")?.to_string_lossy();
        let suffix = unique_suffix()?;
        let candidate = parent.join(format!(".{stem}.update-{suffix}"));
        let backup = parent.join(format!(".{stem}.previous-{suffix}"));
        if target.kind == InstallKind::MacBundle {
            successful(background_command("ditto").arg(&prepared.source).arg(&candidate))?;
        } else {
            let mut output = File::options()
                .write(true)
                .create_new(true)
                .open(&candidate)
                .context("cannot write to the installation folder; move wessh to a folder you own before updating")?;
            std::io::copy(&mut File::open(&prepared.source)?, &mut output)?;
            output.sync_all()?;
            executable(&candidate)?;
        }
        let binary = if target.kind == InstallKind::MacBundle { candidate.join("Contents/MacOS/wessh") } else { candidate.clone() };
        let mut probe = background_command(binary);
        probe.arg("--version");
        clean_appimage_environment(&mut probe);
        if target.kind == InstallKind::AppImage {
            probe.env("APPIMAGE_EXTRACT_AND_RUN", "1");
        }
        let output = probe.output().context("the new application could not start; the installed app is unchanged")?;
        ensure!(
            output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == format!("wessh {}", prepared.update.version),
            "the new application did not pass its version check; the installed app is unchanged"
        );
        let script = prepared.workspace.join("apply.sh");
        fs::write(&script, UNIX_HELPER)?;
        let mut command = background_command("/bin/sh");
        command
            .arg(script)
            .arg(std::process::id().to_string())
            .arg(candidate)
            .arg(&target.path)
            .arg(backup)
            .arg(if target.kind == InstallKind::MacBundle { "bundle" } else { "binary" })
            .arg(if restart { "yes" } else { "no" })
            .args(std::env::args_os().skip(1));
        clean_appimage_environment(&mut command);
        command
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .current_dir(&prepared.workspace)
        .spawn()
        .context("could not start the update helper; the installed app has not changed")?;
    Ok(())
}

#[derive(Serialize)]
struct WindowsConfig<'a> {
    pid: u32,
    installer: bool,
    source: &'a Path,
    target: &'a Path,
    restart: bool,
    arguments: Vec<String>,
}

/// Mounted `AppImage` paths disappear when the old process exits; helpers and the new app must
/// use the host environment rather than libraries, curl, or certificates from that mount.
fn clean_appimage_environment(command: &mut Command) {
    let Some(appdir) = std::env::var_os("APPDIR").map(PathBuf::from) else { return };
    for key in ["PATH", "LD_LIBRARY_PATH"] {
        if let Some(value) = std::env::var_os(key) {
            let paths = std::env::split_paths(&value).filter(|path| !path.starts_with(&appdir));
            if let Ok(value) = std::env::join_paths(paths) {
                command.env(key, value);
            }
        }
    }
    for key in ["CURL_CA_BUNDLE", "XCOMPOSEFILE"] {
        if std::env::var_os(key).is_some_and(|value| Path::new(&value).starts_with(&appdir)) {
            command.env_remove(key);
        }
    }
    for key in ["APPIMAGE", "APPDIR", "OWD"] {
        command.env_remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!("wessh-update-test-{}", unique_suffix().unwrap()));
        fs::create_dir(&path).unwrap();
        path
    }

    #[cfg(unix)]
    fn unix_helper(folder: &Path, pid: u32) -> Command {
        let script = folder.join("apply.sh");
        fs::write(&script, UNIX_HELPER).unwrap();
        let mut command = background_command("/bin/sh");
        command
            .arg(script)
            .arg(pid.to_string())
            .arg(folder.join("new app"))
            .arg(folder.join("wessh $ ' app"))
            .arg(folder.join("previous app"))
            .args(["binary", "no"]);
        command
    }

    #[cfg(unix)]
    #[test]
    fn unix_update_waits_for_exit_replaces_only_app_and_keeps_backup() {
        let folder = directory();
        fs::write(folder.join("wessh $ ' app"), "old").unwrap();
        fs::write(folder.join("new app"), "new").unwrap();
        fs::write(folder.join("model.onnx"), "cached model").unwrap();
        let mut parent = background_command("/bin/sh").args(["-c", "sleep 10"]).spawn().unwrap();
        let helper = unix_helper(&folder, parent.id()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        assert_eq!(fs::read(folder.join("wessh $ ' app")).unwrap(), b"old");
        parent.kill().unwrap();
        parent.wait().unwrap();
        let result = helper.wait_with_output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        assert_eq!(fs::read(folder.join("wessh $ ' app")).unwrap(), b"new");
        assert_eq!(fs::read(folder.join("previous app")).unwrap(), b"old");
        assert_eq!(fs::read(folder.join("model.onnx")).unwrap(), b"cached model");
        fs::remove_dir_all(folder).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn unix_failed_replacement_restores_previous_app() {
        let folder = directory();
        fs::write(folder.join("wessh $ ' app"), "old").unwrap();
        let mut parent = background_command("/bin/sh").args(["-c", "exit 0"]).spawn().unwrap();
        parent.wait().unwrap();
        let result = unix_helper(&folder, parent.id()).output().unwrap();
        assert!(!result.status.success());
        assert_eq!(fs::read(folder.join("wessh $ ' app")).unwrap(), b"old");
        assert!(!folder.join("previous app").exists());
        fs::remove_dir_all(folder).unwrap();
    }

    #[cfg(windows)]
    fn windows_helper(folder: &Path) -> Command {
        let script = folder.join("apply.ps1");
        fs::write(&script, WINDOWS_HELPER).unwrap();
        let config = serde_json::json!({ "pid": 987654321, "installer": false,
            "source": folder.join("new"), "target": folder.join("installed"), "restart": false, "arguments": [] });
        let config_path = folder.join("config.json");
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let mut command = background_command("powershell.exe");
        command.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(script).arg(config_path);
        command
    }

    #[cfg(windows)]
    #[test]
    fn windows_portable_update_rolls_back_both_binaries_if_second_copy_fails() {
        let folder = directory();
        for name in ["installed", "new"] {
            fs::create_dir(folder.join(name)).unwrap();
        }
        for name in ["wessh.exe", "wessh-cli.exe"] {
            fs::write(folder.join("installed").join(name), "old").unwrap();
        }
        fs::write(folder.join("new/wessh.exe"), "new").unwrap();
        windows_helper(&folder).output().unwrap();
        for name in ["wessh.exe", "wessh-cli.exe"] {
            assert_eq!(fs::read(folder.join("installed").join(name)).unwrap(), b"old");
        }
        fs::remove_dir_all(folder).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_portable_update_replaces_gui_and_cli_and_preserves_cache() {
        let folder = directory();
        for name in ["installed", "new"] {
            fs::create_dir(folder.join(name)).unwrap();
        }
        for name in ["wessh.exe", "wessh-cli.exe"] {
            fs::write(folder.join("installed").join(name), "old").unwrap();
            fs::write(folder.join("new").join(name), "new").unwrap();
        }
        fs::write(folder.join("installed/model.onnx"), "cached model").unwrap();
        let result = windows_helper(&folder).output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        for name in ["wessh.exe", "wessh-cli.exe"] {
            assert_eq!(fs::read(folder.join("installed").join(name)).unwrap(), b"new");
        }
        assert_eq!(fs::read(folder.join("installed/model.onnx")).unwrap(), b"cached model");
        fs::remove_dir_all(folder).unwrap();
    }
}
