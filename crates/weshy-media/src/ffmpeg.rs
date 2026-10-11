use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Overrides which ffmpeg is run.
const FFMPEG_VAR: &str = "WESHY_FFMPEG";
const EXE: &str = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };

/// Where `weshy setup` puts its own ffmpeg on machines that have none.
#[must_use]
pub fn bundled_ffmpeg() -> PathBuf {
    weshy_core::assets::data_dir().join("ffmpeg").join(EXE)
}

/// An ffmpeg on `PATH`, found without running anything.
#[must_use]
pub fn system_ffmpeg() -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|dir| dir.join(EXE)).find(|path| path.is_file())
}

/// The ffmpeg weshy runs: `$WESHY_FFMPEG`, else the one `setup` installed, else the one on
/// `PATH`; `None` when there is none.
#[must_use]
pub fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(program) = std::env::var_os(FFMPEG_VAR) {
        return Some(program.into());
    }
    Some(bundled_ffmpeg()).filter(|path| path.is_file()).or_else(system_ffmpeg)
}

/// [`find_ffmpeg`] without any arguments (plain `ffmpeg` when there is none, so the error names it).
pub fn ffmpeg_program() -> Command {
    Command::new(find_ffmpeg().as_deref().unwrap_or(Path::new(EXE)))
}

/// [`ffmpeg_program`] with quiet, non-interactive defaults.
pub fn ffmpeg() -> Command {
    let mut cmd = ffmpeg_program();
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]).stderr(Stdio::inherit());
    cmd
}

/// Arguments describing raw BGR frames on stdin.
pub fn raw_bgr_input(cmd: &mut Command, width: usize, height: usize, fps: f64) {
    cmd.args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-"]);
}

pub fn spawn(cmd: &mut Command) -> Result<Child> {
    cmd.spawn().map_err(|source| Error::Spawn { program: cmd.get_program().to_string_lossy().into_owned(), source })
}
