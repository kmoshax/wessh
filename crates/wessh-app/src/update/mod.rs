//! Updates from this repository's stable GitHub releases; never touches runtime/model caches
//! (or the TensorRT for RTX kernel cache) in the data directory.

mod apply;

use crate::install::fetch::{self, Download, Transfer};
use crate::install::platform::{Arch, Os, Target};
use anyhow::{Context, Result, bail, ensure};
pub use apply::{Prepared, install_and_restart};
use semver::Version;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

pub const CURRENT: &str = env!("CARGO_PKG_VERSION");
const REPO: &str = "kmoshax/wessh";

#[derive(Clone, Debug)]
pub struct Update {
    pub version: String,
    /// What changed, one line per change, from the release notes.
    pub changes: Vec<String>,
    pub download: Download,
    target: InstallTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum InstallKind {
    WindowsInstaller,
    WindowsPortable,
    AppImage,
    MacBundle,
    Binary,
}

#[derive(Clone, Debug)]
struct InstallTarget {
    kind: InstallKind,
    path: PathBuf,
    platform: Target,
}

impl InstallTarget {
    fn detect() -> Result<Self> {
        let platform = Target::host()?;
        let exe = std::env::current_exe()?.canonicalize()?;
        let appimage = std::env::var_os("APPIMAGE").map(PathBuf::from);
        let (kind, path) = match platform.os {
            Os::Linux if appimage.is_some() => (InstallKind::AppImage, appimage.context("missing AppImage path")?.canonicalize()?),
            Os::Windows => {
                let folder = exe.parent().context("executable has no parent directory")?;
                let kind = if folder.join("unins000.exe").is_file() { InstallKind::WindowsInstaller } else { InstallKind::WindowsPortable };
                (kind, folder.to_path_buf())
            }
            Os::Mac => exe
                .ancestors()
                .find(|path| path.extension().is_some_and(|ext| ext == "app"))
                .map_or_else(|| (InstallKind::Binary, exe.clone()), |bundle| (InstallKind::MacBundle, bundle.to_path_buf())),
            Os::Linux => (InstallKind::Binary, exe),
        };
        Ok(Self { kind, path, platform })
    }

    fn asset_name(&self) -> String {
        let arch = match self.platform.arch {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
        };
        match self.kind {
            InstallKind::WindowsInstaller => "wessh-windows-x64-setup.exe".to_owned(),
            InstallKind::WindowsPortable => "wessh-x86_64-pc-windows-msvc.zip".to_owned(),
            InstallKind::AppImage => format!("wessh-linux-{arch}.AppImage"),
            InstallKind::MacBundle => format!("wessh-macos-{}.dmg", if self.platform.arch == Arch::Aarch64 { "arm64" } else { "x86_64" }),
            InstallKind::Binary => {
                format!("wessh-{arch}-{}.tar.gz", if self.platform.os == Os::Mac { "apple-darwin" } else { "unknown-linux-gnu" })
            }
        }
    }
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    /// The release notes (Markdown); the Release workflow lists the commits as `- subject` lines.
    #[serde(default)]
    body: Option<String>,
    draft: bool,
    prerelease: bool,
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
    size: u64,
    browser_download_url: String,
    digest: Option<String>,
}

/// Check once in the background (or on user request). `None` means already up to date.
pub fn check() -> Result<Option<Update>> {
    let release: Release = fetch::get_json(&format!("https://api.github.com/repos/{REPO}/releases/latest"))?;
    let Some(version) = newer_version(&release, CURRENT)? else { return Ok(None) };
    let target = InstallTarget::detect()?;
    let name = target.asset_name();
    let asset = release.assets.iter().find(|asset| asset.name == name).context("this release has no download for your installation")?;
    let base = format!("https://github.com/{REPO}/releases/download/{}/", release.tag_name);
    ensure!(asset.browser_download_url == format!("{base}{name}"), "unexpected update download URL");
    let sha256 = if let Some(hash) = asset.digest.as_deref().and_then(fetch::normalize_sha256) {
        hash
    } else {
        let sums = release.assets.iter().find(|asset| asset.name == "SHA256SUMS").context("release has no checksums")?;
        ensure!(sums.browser_download_url == format!("{base}SHA256SUMS"), "unexpected checksum URL");
        checksum(&fetch::get_text(&sums.browser_download_url, "text/plain")?, &name)?
    };
    Ok(Some(Update {
        version: version.to_string(),
        changes: changes(release.body.as_deref().unwrap_or_default()),
        download: Download { url: asset.browser_download_url.clone(), file_name: name, size: Some(asset.size), sha256: Some(sha256) },
        target,
    }))
}

fn newer_version(release: &Release, current: &str) -> Result<Option<Version>> {
    if release.draft || release.prerelease {
        return Ok(None);
    }
    let version = Version::parse(release.tag_name.strip_prefix('v').unwrap_or(&release.tag_name)).context("invalid release version")?;
    if !version.pre.is_empty() {
        return Ok(None);
    }
    Ok((version > Version::parse(current)?).then_some(version))
}

/// Changes listed in the update dialog, at most this many (the full notes are on GitHub).
const MAX_CHANGES: usize = 8;

/// The list items of release notes: their `- ` / `* ` lines.
fn changes(notes: &str) -> Vec<String> {
    notes
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- ").or_else(|| line.trim().strip_prefix("* ")))
        .map(|change| change.trim().to_owned())
        .filter(|change| !change.is_empty())
        .take(MAX_CHANGES)
        .collect()
}

fn checksum(manifest: &str, name: &str) -> Result<String> {
    let mut matches = manifest.lines().filter_map(|line| {
        let (hash, file) = line.split_once(char::is_whitespace)?;
        (file.trim().trim_start_matches('*') == name).then(|| fetch::normalize_sha256(hash)).flatten()
    });
    let hash = matches.next().context("update checksum is missing or invalid")?;
    ensure!(matches.next().is_none(), "ambiguous update checksum");
    Ok(hash)
}

/// Tidy up after an earlier update (see [`apply::clean_up`]); call once on start.
pub fn clean_up() {
    let (Ok(exe), Ok(current)) = (std::env::current_exe(), Version::parse(CURRENT)) else { return };
    let installed = std::env::var_os("APPIMAGE").map_or(exe, PathBuf::from);
    apply::clean_up(&installed, &updates_dir(), &current);
}

fn updates_dir() -> PathBuf {
    wessh_core::assets::data_dir().join("updates")
}

/// Download and verify, then stage files. Applying them needs a separate explicit action.
pub fn prepare(update: &Update, cancel: &AtomicBool, progress: &mut dyn FnMut(Transfer)) -> Result<Prepared> {
    let folder = updates_dir().join(&update.version);
    std::fs::create_dir_all(&folder)?;
    let archive = folder.join(&update.download.file_name);
    fetch::download(&update.download, &archive, cancel, progress)?;
    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(fetch::Cancelled.into());
    }
    apply::stage(update, &archive, &folder)
}

/// Fixture for the existing local UI capture workflow; never used for real update actions.
pub fn preview() -> Result<Update> {
    let mut version = Version::parse(CURRENT)?;
    version.patch += 1;
    let target = InstallTarget::detect()?;
    Ok(Update {
        version: version.to_string(),
        changes: vec!["Find network cameras on the local Wi-Fi".to_owned(), "Hold faces the detector misses".to_owned()],
        download: Download { url: String::new(), file_name: target.asset_name(), size: Some(13_000_000), sha256: None },
        target,
    })
}

fn binary_from_archive(archive: &Path, destination: &Path) -> Result<()> {
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::fs::File::open(archive)?));
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.file_name().is_some_and(|name| name == "wessh") && entry.header().entry_type().is_file() {
            std::io::copy(&mut entry, &mut std::fs::File::create(destination)?)?;
            return Ok(());
        }
    }
    bail!("update archive has no wessh executable")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(tag: &str) -> Release {
        Release { tag_name: tag.into(), body: None, draft: false, prerelease: false, assets: Vec::new() }
    }

    #[test]
    fn changes_are_the_notes_list_items() {
        let notes = "## What's new\n\n- Find network cameras\n* Hold missed faces\n-   \n\n**Full changelog**: https://x";
        assert_eq!(changes(notes), ["Find network cameras", "Hold missed faces"]);
        let long = "- change\n".repeat(20);
        assert_eq!(changes(&long).len(), MAX_CHANGES);
    }

    #[test]
    fn versions_compare_numerically_and_skip_old_or_unstable_releases() {
        assert!(newer_version(&release("v0.10.0"), "0.9.9").unwrap().is_some());
        assert!(newer_version(&release("v0.3.1"), "0.3.1").unwrap().is_none());
        assert!(newer_version(&release("v0.3.0"), "0.3.1").unwrap().is_none());
        assert!(newer_version(&release("v0.4.0-rc.1"), "0.3.1").unwrap().is_none());
        assert!(newer_version(&Release { draft: true, ..release("v0.4.0") }, "0.3.1").unwrap().is_none());
        assert!(newer_version(&Release { prerelease: true, ..release("v0.4.0") }, "0.3.1").unwrap().is_none());
        assert!(newer_version(&release("bad"), "0.3.1").is_err());
    }

    #[test]
    fn checksum_requires_an_exact_unique_filename() {
        let hash = "ab".repeat(32);
        assert_eq!(checksum(&format!("{hash}  wessh.zip\n"), "wessh.zip").unwrap(), hash);
        assert_eq!(checksum(&format!("{hash} *wessh.zip\n"), "wessh.zip").unwrap(), hash);
        assert!(checksum(&format!("{hash}  other-wessh.zip\n"), "wessh.zip").is_err());
        assert!(checksum(&format!("{hash}  wessh.zip\n{hash}  wessh.zip\n"), "wessh.zip").is_err());
        assert!(checksum("bad  wessh.zip", "wessh.zip").is_err());
    }

    #[test]
    fn package_matches_the_installation_format() {
        let target = |os, arch, kind| InstallTarget { platform: Target { os, arch }, kind, path: PathBuf::new() };
        assert_eq!(target(Os::Linux, Arch::Aarch64, InstallKind::AppImage).asset_name(), "wessh-linux-aarch64.AppImage");
        assert_eq!(target(Os::Linux, Arch::X86_64, InstallKind::Binary).asset_name(), "wessh-x86_64-unknown-linux-gnu.tar.gz");
        assert_eq!(target(Os::Mac, Arch::Aarch64, InstallKind::MacBundle).asset_name(), "wessh-macos-arm64.dmg");
        assert_eq!(target(Os::Windows, Arch::X86_64, InstallKind::WindowsInstaller).asset_name(), "wessh-windows-x64-setup.exe");
        assert_eq!(target(Os::Windows, Arch::X86_64, InstallKind::WindowsPortable).asset_name(), "wessh-x86_64-pc-windows-msvc.zip");
    }
}
