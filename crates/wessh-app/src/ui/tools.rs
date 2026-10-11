//! The ffmpeg that video files and phone cameras need, downloaded from inside the app the first
//! time one of them is used (live webcam mode never needs it).

use crate::install::progress::human_size;
use crate::install::{Event, Host, Installer, Job, Message, Outcome, Plan, Request};
use crate::ui::components::{Button, LinearProgress, empty_state};
use crate::ui::icons;
use crate::ui::theme::spacing;
use eframe::egui::{Context, Ui};

/// Download size shown before the real one is known (the `imageio-ffmpeg` wheels are 21–31 MB).
const TYPICAL_SIZE: &str = "about 30 MB";

/// Whether ffmpeg is there, and its download when it isn't.
#[derive(Default)]
pub struct VideoTools {
    /// Known to be installed; checked once, then after every download.
    ready: Option<bool>,
    installer: Option<Installer>,
    done: u64,
    size: Option<u64>,
    error: Option<String>,
}

impl VideoTools {
    /// ffmpeg can be run.
    pub fn is_ready(&mut self) -> bool {
        *self.ready.get_or_insert_with(|| wessh_media::find_ffmpeg().is_some())
    }

    fn start(&mut self, ctx: &Context) {
        self.error = None;
        let host = match Host::detect(None) {
            Ok(host) => host,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        let plan = Plan::check(host, &Request::ffmpeg_only());
        let ctx = ctx.clone();
        self.installer = Some(Installer::spawn(plan, Job::Install, move || ctx.request_repaint()));
    }

    fn poll(&mut self) {
        let Some(installer) = &self.installer else { return };
        let mut finished = None;
        for message in installer.poll() {
            match message {
                Message::Event(Event::Resolved { size, partial, .. }) => (self.size, self.done) = (size, partial),
                Message::Event(Event::Bytes { done, .. }) => self.done = done,
                Message::Event(Event::Stage { .. }) => {}
                Message::Finished { outcome, .. } => finished = Some(outcome),
            }
        }
        match finished {
            Some(Outcome::Done) => {
                self.installer = None;
                self.ready = Some(wessh_media::find_ffmpeg().is_some());
            }
            Some(Outcome::Failed(error)) => (self.installer, self.error) = (None, Some(error)),
            Some(Outcome::Cancelled) => self.installer = None,
            None => {}
        }
    }

    /// The download card, for `what` ("Videos", "Phone cameras") that need ffmpeg.
    pub fn ui(&mut self, ui: &mut Ui, what: &str) {
        self.poll();
        let size = self.size.map_or_else(|| TYPICAL_SIZE.to_owned(), human_size);
        let body = match &self.error {
            Some(error) => format!("The download didn't finish: {error}"),
            None => format!("{what} need a small video toolkit (ffmpeg, {size}). It is downloaded once and kept with wessh."),
        };
        let downloading = self.installer.is_some();
        let mut start = false;
        empty_state(ui, icons::DOWNLOAD, "One more download", &body, |ui| {
            if downloading {
                let fraction = self.size.filter(|&s| s > 0).map(|s| self.done as f32 / s as f32);
                ui.add(fraction.map_or_else(LinearProgress::indeterminate, LinearProgress::determinate));
                ui.add_space(spacing::SM);
                ui.label(format!("{} of {size}", human_size(self.done)));
            } else {
                let label = if self.error.is_some() { "Try again" } else { "Download" };
                start = ui.add(Button::filled(label).icon(icons::DOWNLOAD)).clicked();
            }
        });
        if start {
            self.start(ui.ctx());
        }
    }
}
