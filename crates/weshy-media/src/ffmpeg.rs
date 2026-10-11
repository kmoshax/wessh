use crate::{Error, Result};
use std::process::{Child, Command, Stdio};

fn program(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_owned())
}

/// `ffmpeg` (or `$WESHY_FFMPEG`) without any arguments.
pub fn ffmpeg_program() -> Command {
    Command::new(program("WESHY_FFMPEG", "ffmpeg"))
}

/// `ffmpeg` (or `$WESHY_FFMPEG`) with quiet, non-interactive defaults.
pub fn ffmpeg() -> Command {
    let mut cmd = ffmpeg_program();
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]).stderr(Stdio::inherit());
    cmd
}

/// `ffprobe` (or `$WESHY_FFPROBE`).
pub fn ffprobe() -> Command {
    Command::new(program("WESHY_FFPROBE", "ffprobe"))
}

/// Arguments describing raw BGR frames on stdin.
pub fn raw_bgr_input(cmd: &mut Command, width: usize, height: usize, fps: f64) {
    cmd.args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-"]);
}

pub fn spawn(cmd: &mut Command) -> Result<Child> {
    cmd.spawn().map_err(|source| Error::Spawn { program: cmd.get_program().to_string_lossy().into_owned(), source })
}
