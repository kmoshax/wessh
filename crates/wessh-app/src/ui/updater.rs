//! MD3 update dialog. Checking is automatic; downloading and restarting are explicit actions.

use super::components::{Button, IconButton, LinearProgress};
use super::icons;
use super::task::Task;
use super::theme::{self, TypeRole, shape, spacing};
use crate::install::fetch::Transfer;
use crate::install::progress::human_size;
use crate::update::{self, Prepared, Update};
use eframe::egui::{Context, Frame, Id, Margin, Modal, Ui};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

pub struct Updater {
    checking: Task<Result<Option<Update>, String>>,
    downloading: Option<JoinHandle<()>>,
    finished: Arc<Mutex<Option<Result<Prepared, String>>>>,
    update: Option<Update>,
    prepared: Option<Prepared>,
    progress: Arc<Mutex<(u64, bool)>>,
    cancel: Arc<AtomicBool>,
    checked: bool,
    error: Option<String>,
    open: bool,
    preview: bool,
}

impl Updater {
    pub fn new(ctx: &Context) -> Self {
        let preview = std::env::var("WESSH_UI_UPDATER").is_ok_and(|value| value == "preview");
        let mut updater = Self {
            checking: Task::default(),
            downloading: None,
            finished: Arc::default(),
            update: None,
            prepared: None,
            progress: Arc::default(),
            cancel: Arc::default(),
            checked: false,
            error: None,
            open: preview,
            preview,
        };
        if preview {
            updater.update = update::preview().ok();
        } else {
            updater.check(ctx);
        }
        updater
    }

    fn check(&mut self, ctx: &Context) {
        self.error = None;
        self.checking.spawn(ctx, || update::check().map_err(|error| format!("{error:#}")));
    }

    /// The application bar shows an update action when a new release is available.
    pub fn button(&mut self, ui: &mut Ui) {
        let clicked = if self.update.is_some() {
            let label = if self.prepared.is_some() {
                "Update ready"
            } else if self.downloading.is_some() {
                "Downloading update…"
            } else {
                "Update available"
            };
            ui.add(Button::text(label).icon(icons::DOWNLOAD)).clicked()
        } else {
            ui.add(IconButton::new(icons::REFRESH, "Check for updates")).clicked()
        };
        if clicked {
            self.open = true;
            if !self.checking.is_running() && self.update.is_none() {
                self.check(ui.ctx());
            }
        }
    }

    fn poll(&mut self) {
        if let Some(result) = self.checking.poll() {
            self.checked = true;
            match result {
                Ok(update) => self.update = update,
                Err(error) => self.error = Some(error),
            }
        }
        let result = self.finished.lock().take();
        if let Some(result) = result {
            if let Some(thread) = self.downloading.take() {
                let _ = thread.join();
            }
            match result {
                Ok(prepared) => self.prepared = Some(prepared),
                Err(error) if !self.cancel.load(Ordering::Relaxed) => self.error = Some(error),
                Err(_) => {}
            }
        }
    }

    fn download(&mut self, ctx: &Context) {
        let Some(update) = self.update.clone() else { return };
        self.error = None;
        self.cancel = Arc::default();
        self.progress = Arc::default();
        let cancel = self.cancel.clone();
        let progress = self.progress.clone();
        let repaint = ctx.clone();
        let finished = self.finished.clone();
        self.downloading = Some(std::thread::spawn(move || {
            let result = update::prepare(&update, &cancel, &mut |event| {
                let mut state = progress.lock();
                match event {
                    Transfer::Bytes(done) => *state = (done, false),
                    Transfer::Verifying => state.1 = true,
                }
                repaint.request_repaint();
            })
            .map_err(|error| format!("{error:#}"));
            *finished.lock() = Some(result);
            repaint.request_repaint();
        }));
    }

    fn show_error(&self, ui: &mut Ui) {
        if let Some(error) = &self.error {
            let colors = theme::colors(ui.ctx());
            ui.add_space(spacing::MD);
            ui.label(TypeRole::BodyMedium.rich("Couldn't finish the update. Your installed app is unchanged.", colors.error));
            ui.label(TypeRole::BodySmall.rich(error, colors.on_surface_variant));
        }
    }

    fn show_download_info(&self, ui: &mut Ui, update: &Update) {
        let colors = theme::colors(ui.ctx());
        ui.add_space(spacing::MD);
        if let Some(size) = update.download.size {
            ui.label(format!("Download size: {}", human_size(size)));
        }
        if self.cancel.load(Ordering::Relaxed) {
            ui.label("Download paused. Your progress is saved; retry to resume.");
        }
        ui.label(TypeRole::BodySmall.rich("Updates come from GitHub and are verified with SHA-256.", colors.on_surface_variant));
    }

    /// `true` requests a clean app shutdown after the update helper has been started.
    pub fn ui(&mut self, ctx: &Context, busy: bool) -> bool {
        self.poll();
        if !self.open {
            return false;
        }
        let colors = theme::colors(ctx);
        let frame = Frame::new()
            .fill(colors.surface_container_high)
            .corner_radius(shape::EXTRA_LARGE)
            .inner_margin(Margin::same(spacing::XL as i8));
        let mut download = false;
        let mut install = false;
        let mut check = false;
        let mut close = false;
        let modal = Modal::new(Id::new("app-updater")).frame(frame).show(ctx, |ui| {
            ui.set_width(440.0_f32.min(ctx.content_rect().width() - 2.0 * spacing::XL));
            ui.label(TypeRole::HeadlineSmall.rich("Updates", colors.on_surface));
            ui.add_space(spacing::SM);
            ui.label(TypeRole::BodyMedium.rich(format!("Installed version: {}", update::CURRENT), colors.on_surface_variant));
            ui.add_space(spacing::LG);
            if let Some(update) = &self.update {
                ui.label(TypeRole::TitleLarge.rich(format!("wessh {} is available", update.version), colors.on_surface));
                ui.add_space(spacing::SM);
                ui.label("Update the app without downloading your face models or runtime again.");
                if !update.changes.is_empty() {
                    ui.add_space(spacing::MD);
                    ui.label(TypeRole::TitleSmall.rich("What's new", colors.on_surface));
                    for change in &update.changes {
                        ui.label(TypeRole::BodyMedium.rich(format!("•  {change}"), colors.on_surface_variant));
                    }
                }
                if self.prepared.is_some() {
                    ui.add_space(spacing::MD);
                    ui.label("Download verified. Install and restart when you're ready. Save your work first.");
                } else if self.downloading.is_some() {
                    let (done, verifying) = *self.progress.lock();
                    ui.add_space(spacing::LG);
                    ui.add(
                        update
                            .download
                            .size
                            .filter(|&size| size > 0)
                            .map_or_else(LinearProgress::indeterminate, |size| LinearProgress::determinate(done as f32 / size as f32)),
                    );
                    let detail = if self.cancel.load(Ordering::Relaxed) {
                        "Stopping…".to_owned()
                    } else if verifying {
                        "Verifying and preparing the update…".to_owned()
                    } else {
                        download_detail(done, update.download.size)
                    };
                    ui.label(detail);
                } else {
                    self.show_download_info(ui, update);
                }
            } else if self.checking.is_running() {
                ui.add(LinearProgress::indeterminate());
                ui.label("Checking for updates…");
            } else if self.error.is_none() && self.checked {
                ui.label("You're up to date.");
            }
            self.show_error(ui);
            if busy && self.prepared.is_some() {
                ui.label("Finish your current job before restarting.");
            }
            ui.add_space(spacing::XL);
            ui.horizontal(|ui| {
                close = ui.add(Button::text("Later")).clicked();
                if self.downloading.is_some() {
                    if ui.add_enabled(!self.cancel.load(Ordering::Relaxed), Button::outlined("Cancel download")).clicked() {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                } else if self.prepared.is_some() {
                    install = ui.add_enabled(!busy, Button::filled("Install and restart").icon(icons::REFRESH)).clicked();
                } else if self.update.is_some() {
                    let label = if self.error.is_some() { "Retry download" } else { "Download update" };
                    download = ui.add(Button::filled(label).icon(icons::DOWNLOAD)).clicked();
                } else {
                    check = ui.add_enabled(!self.checking.is_running(), Button::filled("Check again")).clicked();
                }
            });
        });
        if close || modal.should_close() {
            self.open = false;
        }
        if !self.preview {
            if download {
                self.download(ctx);
            }
            if check {
                self.check(ctx);
            }
            if install && let Some(prepared) = &self.prepared {
                match update::install_and_restart(prepared) {
                    Ok(()) => return true,
                    Err(error) => self.error = Some(format!("{error:#}")),
                }
            }
        }
        false
    }
}

impl Drop for Updater {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(thread) = self.downloading.take() {
            let _ = thread.join();
        }
    }
}

fn download_detail(done: u64, total: Option<u64>) -> String {
    total.filter(|&size| size > 0).map_or_else(
        || format!("{} downloaded", human_size(done)),
        |size| format!("{} / {} · {:.0}%", human_size(done.min(size)), human_size(size), (done as f64 / size as f64 * 100.0).min(100.0)),
    )
}
