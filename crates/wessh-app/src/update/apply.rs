//! Installing a downloaded update over the running app, then restarting it.
//!
//! A running program can't be overwritten, but it can be renamed — on Windows too — and on Unix a
//! rename atomically replaces the file while the running process keeps the old one open. So the
//! app swaps its own files in place: no helper process waiting for it to exit, no scripts. The
//! Inno Setup install is the exception: its installer runs silently, so Windows keeps the
//! installed version on record, and reopens the app itself. What the swap leaves behind (the
//! previous version, the downloads) is removed by [`clean_up`] on the next start.

use super::{InstallKind, Update, binary_from_archive};
use anyhow::{Context, Result, bail, ensure};
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use wessh_core::process::background_command;

/// Marks what an update set aside: `.<name>.wessh-old-<suffix>` beside the installed files.
const OLD_MARKER: &str = ".wessh-old-";
/// Inno Setup switches for an unattended update; `/relaunch=yes` makes the installer (see
/// `scripts/windows/wessh.iss`) reopen the app once it is done.
const INSTALLER_SWITCHES: [&str; 5] = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/relaunch=yes"];

#[derive(Clone, Debug)]
pub struct Prepared {
    update: Update,
    /// The installer, the new executable, or a folder of new executables / the new bundle.
    source: PathBuf,
}

fn unique_suffix() -> Result<String> {
    Ok(format!("{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()))
}

/// Unpack the downloaded `archive` into `folder`, ready to install.
pub(super) fn stage(update: &Update, archive: &Path, folder: &Path) -> Result<Prepared> {
    let workspace = folder.join(format!("stage-{}", unique_suffix()?));
    fs::create_dir(&workspace)?;
    let source = match update.target.kind {
        InstallKind::WindowsInstaller | InstallKind::AppImage => archive.to_path_buf(),
        InstallKind::WindowsPortable => {
            let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
            for name in WINDOWS_EXECUTABLES {
                let mut entry = zip.by_name(name).with_context(|| format!("update archive is missing {name}"))?;
                ensure!(entry.is_file(), "invalid executable in update archive");
                std::io::copy(&mut entry, &mut File::create(workspace.join(name))?)?;
            }
            workspace
        }
        InstallKind::Binary => {
            let binary = workspace.join("wessh");
            binary_from_archive(archive, &binary)?;
            binary
        }
        InstallKind::MacBundle => stage_bundle(archive, &workspace)?,
    };
    Ok(Prepared { update: update.clone(), source })
}

/// The executables of a Windows install; `wessh-cli.exe` is the console build of the same program.
const WINDOWS_EXECUTABLES: [&str; 2] = ["wessh.exe", "wessh-cli.exe"];

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

/// Install the update and start the new version with this one's arguments; the caller then
/// closes the app. On failure the installed app is left as it was.
pub fn install_and_restart(prepared: &Prepared) -> Result<()> {
    let target = &prepared.update.target;
    let version = &prepared.update.version;
    let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    match target.kind {
        InstallKind::WindowsInstaller => {
            // A GUI program run silently: no window, and it reopens the app when it is done.
            Command::new(&prepared.source).args(INSTALLER_SWITCHES).arg(format!("/DIR={}", target.path.display())).spawn()?;
            return Ok(());
        }
        InstallKind::WindowsPortable => {
            let cli = prepared.source.join("wessh-cli.exe");
            check_version(&cli, version)?;
            let pairs = WINDOWS_EXECUTABLES.map(|name| (prepared.source.join(name), target.path.join(name)));
            replace_all(&pairs)?;
        }
        InstallKind::Binary | InstallKind::AppImage => {
            let candidate = sibling(&target.path, ".wessh-new-")?;
            fs::copy(&prepared.source, &candidate).context("cannot write to the installation folder; move wessh to a folder you own")?;
            executable(&candidate)?;
            let checked = check_version(&candidate, version);
            if let Err(e) = checked.and_then(|()| fs::rename(&candidate, &target.path).map_err(Into::into)) {
                let _ = fs::remove_file(&candidate);
                return Err(e);
            }
        }
        InstallKind::MacBundle => {
            check_version(&prepared.source.join("Contents/MacOS/wessh"), version)?;
            // Copied beside the installed bundle first: the staged one may be on another volume.
            let candidate = sibling(&target.path, ".wessh-new-")?;
            successful(background_command("ditto").arg(&prepared.source).arg(&candidate))?;
            let old = sibling(&target.path, OLD_MARKER)?;
            fs::rename(&target.path, &old).context("cannot write to the installation folder")?;
            if let Err(e) = fs::rename(&candidate, &target.path) {
                fs::rename(&old, &target.path)?;
                let _ = fs::remove_dir_all(&candidate);
                return Err(e.into());
            }
        }
    }
    let mut app = if target.kind == InstallKind::MacBundle {
        let mut open = Command::new("/usr/bin/open");
        open.arg("-n").arg(&target.path).arg("--args");
        open
    } else {
        let exe = if target.kind == InstallKind::WindowsPortable { target.path.join("wessh.exe") } else { target.path.clone() };
        Command::new(exe)
    };
    clean_appimage_environment(&mut app);
    app.args(arguments).spawn().context("the update is installed, but the new version could not be started")?;
    Ok(())
}

/// Replace each `(new, installed)` file, renaming the installed one aside first (Windows can't
/// overwrite a running executable); if any step fails, every file is put back.
fn replace_all(pairs: &[(PathBuf, PathBuf)]) -> Result<()> {
    let mut done: Vec<(&Path, PathBuf)> = Vec::new();
    for (new, installed) in pairs {
        let step = (|| -> Result<PathBuf> {
            let old = sibling(installed, OLD_MARKER)?;
            if installed.exists() {
                fs::rename(installed, &old).context("cannot write to the installation folder")?;
            }
            if let Err(e) = fs::copy(new, installed) {
                let _ = fs::rename(&old, installed);
                return Err(e.into());
            }
            Ok(old)
        })();
        match step {
            Ok(old) => done.push((installed, old)),
            Err(e) => {
                for (installed, old) in done.into_iter().rev() {
                    let _ = fs::remove_file(installed);
                    let _ = fs::rename(old, installed);
                }
                return Err(e);
            }
        }
    }
    Ok(())
}

/// A hidden name beside `path`, unique to this update.
fn sibling(path: &Path, marker: &str) -> Result<PathBuf> {
    let name = path.file_name().context("installation has no file name")?.to_string_lossy();
    Ok(path.with_file_name(format!(".{name}{marker}{}", unique_suffix()?)))
}

/// Run `program --version` and make sure it is `version`, so a broken download never replaces a
/// working app.
fn check_version(program: &Path, version: &str) -> Result<()> {
    let mut probe = background_command(program);
    probe.arg("--version");
    clean_appimage_environment(&mut probe);
    probe.env("APPIMAGE_EXTRACT_AND_RUN", "1");
    let output = probe.output().context("the new version could not start; the installed app is unchanged")?;
    let reported = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || reported.trim() != format!("wessh {version}") {
        bail!("the new version failed its self-check ({}); the installed app is unchanged", reported.trim());
    }
    Ok(())
}

#[cfg(unix)]
fn executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    /// rwxr-xr-x
    const EXECUTABLE: u32 = 0o755;
    fs::set_permissions(path, fs::Permissions::from_mode(EXECUTABLE))?;
    Ok(())
}

#[cfg(not(unix))]
fn executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Remove what earlier updates left: the previous version set aside beside the installed files
/// (it can't be deleted while it runs, but it has exited by the next start), and downloads of
/// versions no newer than this one. Best effort: nothing here may stop the app from starting.
pub fn clean_up(installed: &Path, updates: &Path, current: &semver::Version) {
    if let Some(dir) = installed.parent()
        && let Ok(entries) = fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().contains(OLD_MARKER) {
                let path = entry.path();
                let _ = if path.is_dir() { fs::remove_dir_all(path) } else { fs::remove_file(path) };
            }
        }
    }
    if let Ok(entries) = fs::read_dir(updates) {
        for entry in entries.flatten() {
            let stale = semver::Version::parse(&entry.file_name().to_string_lossy()).is_ok_and(|version| version <= *current);
            if stale {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// Mounted `AppImage` paths disappear when the old process exits; the new app must use the host
/// environment rather than libraries, curl, or certificates from that mount.
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

    #[test]
    fn replacing_swaps_every_file_and_keeps_the_rest() {
        let folder = directory();
        let pairs: Vec<_> = ["a.exe", "b.exe"].iter().map(|n| (folder.join(format!("new-{n}")), folder.join(n))).collect();
        for (new, installed) in &pairs {
            fs::write(new, "new").unwrap();
            fs::write(installed, "old").unwrap();
        }
        fs::write(folder.join("model.onnx"), "cached model").unwrap();
        replace_all(&pairs).unwrap();
        for (_, installed) in &pairs {
            assert_eq!(fs::read(installed).unwrap(), b"new");
        }
        assert_eq!(fs::read(folder.join("model.onnx")).unwrap(), b"cached model");

        // The set-aside versions go on the next start, along with stale downloads.
        let updates = folder.join("updates");
        for version in ["0.4.0", "0.4.1", "0.5.0"] {
            fs::create_dir_all(updates.join(version)).unwrap();
        }
        clean_up(&pairs[0].1, &updates, &semver::Version::new(0, 4, 1));
        let mut left: Vec<String> = fs::read_dir(&folder).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, ["a.exe", "b.exe", "model.onnx", "new-a.exe", "new-b.exe", "updates"]);
        assert!(updates.join("0.5.0").exists() && !updates.join("0.4.1").exists() && !updates.join("0.4.0").exists());
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn a_failed_replacement_puts_everything_back() {
        let folder = directory();
        fs::write(folder.join("new-a.exe"), "new").unwrap();
        for name in ["a.exe", "b.exe"] {
            fs::write(folder.join(name), "old").unwrap();
        }
        // The second new file is missing, so its copy fails after the first was replaced.
        let pairs = [(folder.join("new-a.exe"), folder.join("a.exe")), (folder.join("new-b.exe"), folder.join("b.exe"))];
        assert!(replace_all(&pairs).is_err());
        for name in ["a.exe", "b.exe"] {
            assert_eq!(fs::read(folder.join(name)).unwrap(), b"old");
        }
        assert_eq!(fs::read_dir(&folder).unwrap().count(), 3, "nothing set aside is left behind");
        fs::remove_dir_all(folder).unwrap();
    }
}
