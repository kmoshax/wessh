//! Native "open image" dialog without a GUI-toolkit dependency.
//!
//! Runs the desktop's file chooser (`zenity`, else `kdialog`) on a helper thread and
//! hands the chosen path back on a later frame. Where neither exists the picker
//! reports itself unavailable and the UI falls back to drag-and-drop.

use parking_lot::Mutex;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

const TITLE: &str = "Choose a face photo";
const IMAGE_GLOBS: &str = "*.jpg *.jpeg *.png *.webp *.bmp";

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

    /// Open the dialog; `repaint` is called once a file was chosen.
    pub fn open(&self, repaint: impl Fn() + Send + 'static) {
        let mut busy = self.busy.lock();
        if *busy {
            return;
        }
        *busy = true;
        let (result, busy) = (self.result.clone(), self.busy.clone());
        std::thread::spawn(move || {
            *result.lock() = run_dialog();
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
            (true, true) => "or click to browse · JPG, PNG, WebP",
            (true, false) => "Click to browse · JPG, PNG, WebP",
            (false, _) => "JPG, PNG or WebP with one clear face",
        }
    }
}

/// winit only delivers file drops on X11, macOS and Windows; native Wayland sessions get none.
#[must_use]
pub fn drag_and_drop_supported() -> bool {
    !(cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some())
}

fn program_exists(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(name).is_file()))
}

fn run_dialog() -> Option<PathBuf> {
    let mut command = if program_exists("zenity") {
        let mut c = Command::new("zenity");
        c.args(["--file-selection", "--title", TITLE, "--file-filter", &format!("Images | {IMAGE_GLOBS}")]);
        c
    } else {
        let mut c = Command::new("kdialog");
        c.args(["--title", TITLE, "--getopenfilename", ".", &format!("Images ({IMAGE_GLOBS})")]);
        c
    };
    let output = command.output().ok()?;
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (output.status.success() && !path.is_empty()).then(|| PathBuf::from(path))
}
