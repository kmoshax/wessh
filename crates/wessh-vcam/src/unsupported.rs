//! Off Windows there is no virtual camera to install or start; these keep callers free of `cfg`.

use crate::{Error, Result, Status};
use std::path::{Path, PathBuf};

#[must_use]
pub const fn status() -> Status {
    Status::Unsupported
}

pub fn install(_dll: &Path) -> Result<PathBuf> {
    Err(Error::Unsupported)
}

pub fn uninstall() -> Result<()> {
    Err(Error::Unsupported)
}

#[must_use]
pub const fn is_elevated() -> bool {
    false
}

pub fn run_elevated(_program: &Path, _arguments: &str) -> Result<u32> {
    Err(Error::Unsupported)
}
