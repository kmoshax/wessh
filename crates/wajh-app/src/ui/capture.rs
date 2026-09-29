//! Startup overrides and single-frame capture, for docs and design review.
//!
//! Environment variables (all optional):
//! - `WAJH_UI_CAPTURE=<file.png>`: once the engine has loaded, save one frame and quit.
//! - `WAJH_UI_CAPTURE_AT=<seconds>`: capture at a fixed time instead (e.g. to show loading).
//! - `WAJH_UI_THEME=light|dark`: start in the given brightness.
//! - `WAJH_UI_SCREEN=live|image|video`: start on the given destination.
//! - `WAJH_UI_SOURCE=<image>`: preload a source face.
//! - `WAJH_UI_TARGET=<file>`: preload the photo (Image screen) or video (Video screen) to swap.
//! - `WAJH_UI_OUTPUT=<file>`: where the Video screen writes.
//! - `WAJH_UI_RENDER=1`: start rendering the video as soon as it and the face are ready.
//! - `WAJH_UI_LOOPBACK=missing`: show the Live screen as if v4l2loopback were not loaded.
//! - `WAJH_UI_ONBOARDING=welcome|download|done`: show that onboarding page even if everything
//!   is installed (with a temporary `WAJH_HOME`, onboarding shows by itself). While onboarding
//!   shows, `WAJH_UI_CAPTURE` saves the page 2 s after start unless `WAJH_UI_CAPTURE_AT` is set.
//! - `WAJH_UI_PAUSE_AT=<seconds>`: pause the onboarding download at that time.

use super::screens::Screen;
use super::screens::onboarding::Step;
use crate::live::Phase;
use eframe::egui::{self, Context, Event, UserData, ViewportCommand};
use std::path::{Path, PathBuf};

/// Seconds to wait after loading so queued commands finish and animations settle.
const SETTLE_SECONDS: f64 = 2.0;

/// Startup overrides read from the environment.
#[derive(Default)]
pub struct DevOptions {
    pub capture: Option<Capture>,
    pub light: bool,
    pub screen: Screen,
    pub source: Option<PathBuf>,
    pub target: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub render: bool,
    pub loopback_missing: bool,
    pub onboarding: Option<Step>,
}

impl DevOptions {
    /// Read the `WAJH_UI_*` variables.
    #[must_use]
    pub fn from_env() -> Self {
        let var = |name| std::env::var(name).unwrap_or_default().to_ascii_lowercase();
        let screen = match var("WAJH_UI_SCREEN").as_str() {
            "image" => Screen::Image,
            "video" => Screen::Video,
            _ => Screen::Live,
        };
        Self {
            capture: std::env::var_os("WAJH_UI_CAPTURE").map(|path| Capture {
                path: path.into(),
                at: std::env::var("WAJH_UI_CAPTURE_AT").ok().and_then(|s| s.parse().ok()),
                ready_at: None,
                requested: false,
            }),
            light: var("WAJH_UI_THEME") == "light",
            screen,
            source: std::env::var_os("WAJH_UI_SOURCE").map(PathBuf::from),
            target: std::env::var_os("WAJH_UI_TARGET").map(PathBuf::from),
            output: std::env::var_os("WAJH_UI_OUTPUT").map(PathBuf::from),
            render: var("WAJH_UI_RENDER") == "1",
            loopback_missing: var("WAJH_UI_LOOPBACK") == "missing",
            onboarding: Step::parse(&var("WAJH_UI_ONBOARDING")),
        }
    }
}

/// Saves a single frame to a PNG, then closes the app.
pub struct Capture {
    path: PathBuf,
    /// Fixed capture time in seconds; otherwise wait for the engine to load.
    at: Option<f64>,
    ready_at: Option<f64>,
    requested: bool,
}

impl Capture {
    /// Drive the capture; call once per frame.
    pub fn update(&mut self, ctx: &Context, phase: &Phase) {
        ctx.request_repaint();
        let now = ctx.input(|i| i.time);
        if *phase != Phase::Loading && self.ready_at.is_none() {
            self.ready_at = Some(now);
        }
        let due = match self.at {
            Some(at) => now >= at,
            None => self.ready_at.is_some_and(|ready| now - ready >= SETTLE_SECONDS),
        };
        if !self.requested && due {
            self.requested = true;
            ctx.send_viewport_cmd(ViewportCommand::Screenshot(UserData::default()));
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = shot {
            if let Err(error) = save(&image, &self.path) {
                eprintln!("capture failed: {error:#}");
            }
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
    }
}

fn save(image: &egui::ColorImage, path: &Path) -> anyhow::Result<()> {
    let [width, height] = image.size;
    let pixels: Vec<u8> = image.pixels.iter().flat_map(egui::Color32::to_srgba_unmultiplied).collect();
    let buffer =
        image::RgbaImage::from_raw(width as u32, height as u32, pixels).ok_or_else(|| anyhow::anyhow!("screenshot size mismatch"))?;
    buffer.save(path)?;
    Ok(())
}
