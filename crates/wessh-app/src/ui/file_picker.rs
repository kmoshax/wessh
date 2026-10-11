//! Native open / save dialogs without a GUI-toolkit dependency.
//!
//! Runs the desktop's file chooser (`zenity`, else `kdialog`) on a helper thread and
//! hands the chosen path back on a later frame. Where neither exists the picker
//! reports itself unavailable and the UI falls back to drag-and-drop or a path field.

use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const IMAGE_GLOBS: &str = "*.jpg *.jpeg *.png";
const VIDEO_GLOBS: &str = "*.mp4 *.mov *.mkv *.webm *.avi *.m4v";

/// Extensions accepted as photos (the formats the `image` crate is built with).
pub const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// Which dialog to show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dialog {
    /// Choose a face photo (source).
    SourceImage,
    /// Choose a photo to swap faces in.
    TargetImage,
    /// Choose a video to swap faces in.
    TargetVideo,
    /// Choose where to save, starting from `default`.
    Save { title: &'static str, default: PathBuf },
}

impl Dialog {
    fn title(&self) -> &'static str {
        match self {
            Self::SourceImage => "Choose a face photo",
            Self::TargetImage => "Choose a photo",
            Self::TargetVideo => "Choose a video",
            Self::Save { title, .. } => title,
        }
    }

    fn filter(&self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::SourceImage | Self::TargetImage => Some(("Images", IMAGE_GLOBS)),
            Self::TargetVideo => Some(("Videos", VIDEO_GLOBS)),
            Self::Save { .. } => None,
        }
    }
}

/// An asynchronous single-file picker.
#[derive(Default)]
pub struct FilePicker {
    result: Arc<Mutex<Option<PathBuf>>>,
    busy: Arc<Mutex<bool>>,
}

impl FilePicker {
    /// Whether a supported dialog program is installed.
    #[must_use]
    pub fn available() -> bool {
        cfg!(target_os = "linux") && (program_exists("zenity") || program_exists("kdialog"))
    }

    /// Whether a dialog is currently open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        *self.busy.lock()
    }

    /// Open `dialog`; `repaint` is called once it closes.
    pub fn open(&self, dialog: Dialog, repaint: impl Fn() + Send + 'static) {
        let mut busy = self.busy.lock();
        if *busy {
            return;
        }
        *busy = true;
        let (result, busy) = (self.result.clone(), self.busy.clone());
        std::thread::spawn(move || {
            *result.lock() = run_dialog(&dialog);
            *busy.lock() = false;
            repaint();
        });
    }

    /// The chosen file, once.
    #[must_use]
    pub fn take(&self) -> Option<PathBuf> {
        self.result.lock().take()
    }
}

/// The ways a source image can reach the app on this system, and the matching copy.
#[derive(Clone, Copy, Debug)]
pub struct Intake {
    /// A native file dialog is available (and not already open).
    pub pick: bool,
    /// The windowing backend delivers file drops.
    pub drop: bool,
}

impl Intake {
    /// Probe the current system; `picker_busy` hides the picker while a dialog is open.
    #[must_use]
    pub fn detect(picker_busy: bool) -> Self {
        Self { pick: FilePicker::available() && !picker_busy, drop: drag_and_drop_supported() }
    }

    /// Neither a dialog nor drag-and-drop: the UI must offer a path field.
    #[must_use]
    pub fn needs_path_field(self) -> bool {
        !self.pick && !self.drop
    }

    /// Call to action for an empty source slot.
    #[must_use]
    pub fn title(self) -> &'static str {
        if self.drop { "Drop a face photo" } else { "Choose a face photo" }
    }

    /// Secondary line under [`Self::title`].
    #[must_use]
    pub fn hint(self) -> &'static str {
        match (self.pick, self.drop) {
            (true, true) => "or click to browse · JPG, PNG",
            (true, false) => "Click to browse · JPG, PNG",
            (false, _) => "JPG or PNG with one clear face",
        }
    }
}

/// winit only delivers file drops on X11, macOS and Windows; native Wayland sessions get none.
#[must_use]
pub fn drag_and_drop_supported() -> bool {
    !(cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some())
}

/// Whether `path` has one of `extensions` (case-insensitive).
#[must_use]
pub fn has_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// `<dir>/<stem>-wessh.<extension>` beside `target`.
#[must_use]
pub fn sibling_output(target: &Path, extension: &str) -> PathBuf {
    let stem = target.file_stem().map_or_else(|| "output".into(), |s| s.to_string_lossy());
    target.with_file_name(format!("{stem}-wessh.{extension}"))
}

/// Show `path`'s folder in the desktop file manager.
pub fn reveal(path: &Path) {
    let folder = if path.is_dir() { path } else { path.parent().unwrap_or(Path::new(".")) };
    let program = match std::env::consts::OS {
        "macos" => "open",
        "windows" => "explorer",
        _ => "xdg-open",
    };
    let _ = Command::new(program).arg(folder).spawn();
}

fn program_exists(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(name).is_file()))
}

fn run_dialog(dialog: &Dialog) -> Option<PathBuf> {
    let title = dialog.title();
    let mut command = if program_exists("zenity") {
        let mut c = Command::new("zenity");
        c.args(["--file-selection", "--title", title]);
        if let Some((name, globs)) = dialog.filter() {
            c.args(["--file-filter", &format!("{name} | {globs}")]);
        }
        if let Dialog::Save { default, .. } = dialog {
            c.args(["--save", "--confirm-overwrite"]).arg(format!("--filename={}", default.display()));
        }
        c
    } else {
        let mut c = Command::new("kdialog");
        c.args(["--title", title]);
        if let Dialog::Save { default, .. } = dialog {
            c.arg("--getsavefilename").arg(default);
        } else {
            let (name, globs) = dialog.filter().unwrap_or(("Files", "*"));
            c.args(["--getopenfilename", ".", &format!("{name} ({globs})")]);
        }
        c
    };
    let output = command.output().ok()?;
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (output.status.success() && !path.is_empty()).then(|| PathBuf::from(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_sits_beside_the_target() {
        assert_eq!(sibling_output(Path::new("/a/clip.mov"), "mp4"), PathBuf::from("/a/clip-wessh.mp4"));
        assert!(has_extension(Path::new("x.JPG"), &IMAGE_EXTENSIONS));
        assert!(!has_extension(Path::new("x.mp4"), &IMAGE_EXTENSIONS));
    }
}
