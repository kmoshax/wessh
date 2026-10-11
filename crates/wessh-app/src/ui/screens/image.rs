//! The Image screen: swap faces in a photo with the shared source face and settings, compare
//! before and after, and save. The swap re-runs on its own when the face or settings change.

use super::layout::{centered, column, pane, path_field, stat_chips};
use super::live::{Action, Frame, upload};
use crate::jobs::image::display;
use crate::jobs::{Finished, Job, JobHandle, JobRunner, Status};
use crate::live::{Image, Phase};
use crate::ui::components::compare::Compare;
use crate::ui::components::{Button, Card, CircularProgress, LinearProgress, empty_state, paint};
use crate::ui::file_picker::{Dialog, FilePicker, IMAGE_EXTENSIONS, has_extension, sibling_output};
use crate::ui::icons;
use crate::ui::task::Task;
use crate::ui::theme::{self, TypeRole, size, spacing};
use eframe::egui::{Align, Context, Label, Layout, Rect, TextureHandle, Ui, UiBuilder, Vec2, pos2, vec2};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use wessh_core::{Settings, Stats};

/// Seconds settings must stay unchanged before the photo is swapped again (slider drags).
const DEBOUNCE: f64 = 0.25;
const TOOLBAR_HEIGHT: f32 = 56.0;
/// Height of the stat-chip row under the photo (chip height plus a gap).
const CHIP_ROW: f32 = crate::ui::components::chip::HEIGHT + spacing::MD;
/// Inset of the toolbar, viewer and overlays from the pane edges.
const INSET: f32 = spacing::LG;
const EMPTY_WIDTH: f32 = 420.0;
const NOTICE_WIDTH: f32 = 360.0;
/// MD3 scrim opacity, over the photo while there is no result.
const SCRIM_OPACITY: f32 = 0.32;
/// Where the comparison divider starts.
const INITIAL_SPLIT: f32 = 0.5;
/// Format used when the target's own format can't be written.
const FALLBACK_EXTENSION: &str = "png";

/// What a swap was (or will be) computed from.
#[derive(Clone, Debug, PartialEq)]
struct Request {
    source: PathBuf,
    target: PathBuf,
    settings: Settings,
}

/// State of the Image screen.
pub struct ImageScreen {
    target: Option<PathBuf>,
    target_size: Option<(u32, u32)>,
    path_text: String,
    picker: FilePicker,
    save_picker: FilePicker,
    /// The target photo, loaded for display before any swap.
    loading: Task<Result<Image, String>>,
    before: Option<TextureHandle>,
    after: Option<TextureHandle>,
    result: Option<Arc<RgbImage>>,
    stats: Option<Stats>,
    job: Option<JobHandle>,
    /// The request the running job or the shown result was made from.
    submitted: Option<Request>,
    /// The latest wanted request and when it last changed (for debouncing).
    wanted: Option<(Request, f64)>,
    split: f32,
    saving: Task<Result<PathBuf, String>>,
    save_path: Option<PathBuf>,
    failed: Option<String>,
}

impl Default for ImageScreen {
    fn default() -> Self {
        Self {
            target: None,
            target_size: None,
            path_text: String::new(),
            picker: FilePicker::default(),
            save_picker: FilePicker::default(),
            loading: Task::default(),
            before: None,
            after: None,
            result: None,
            stats: None,
            job: None,
            submitted: None,
            wanted: None,
            split: INITIAL_SPLIT,
            saving: Task::default(),
            save_path: None,
            failed: None,
        }
    }
}

impl ImageScreen {
    /// Swap faces in `path` from now on.
    pub fn set_target(&mut self, ctx: &Context, path: PathBuf) {
        let load = path.clone();
        self.loading.spawn(ctx, move || {
            image::open(&load).map(|image| display(&image.to_rgb8())).map_err(|e| format!("Could not open {}: {e}", load.display()))
        });
        *self = Self { target: Some(path), loading: std::mem::take(&mut self.loading), split: self.split, ..Self::default() };
    }

    /// Draw the screen into `rect`; jobs go to `runner`.
    pub fn ui(&mut self, ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>, runner: &JobRunner) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx, frame);
        self.schedule(&ctx, frame, runner);
        pane(ui, rect);
        if self.target.is_none() {
            self.empty(ui, rect);
            return;
        }
        let toolbar = Rect::from_min_size(rect.min + Vec2::splat(INSET), vec2(rect.width() - 2.0 * INSET, TOOLBAR_HEIGHT));
        self.toolbar(ui, toolbar);
        let viewer = Rect::from_min_max(pos2(rect.left() + INSET, toolbar.bottom() + spacing::SM), rect.max - Vec2::splat(INSET));
        self.viewer(ui, viewer, frame);
    }

    /// Collect finished work: picked files, the loaded target, the swap, saves.
    fn poll(&mut self, ctx: &Context, frame: &mut Frame<'_>) {
        if let Some(path) = self.picker.take() {
            self.set_target(ctx, path);
        }
        if let Some(path) = self.save_picker.take() {
            self.save(ctx, path);
        }
        match self.loading.poll() {
            Some(Ok(image)) => {
                self.target_size = Some((image.width as u32, image.height as u32));
                if self.result.is_none() {
                    upload(ctx, &mut self.before, "image-before", &image);
                }
            }
            Some(Err(error)) => self.fail(frame, error),
            None => {}
        }
        if let Some(job) = &self.job {
            match job.status() {
                Status::Done(Finished::Image(stats)) => {
                    if let Some(swapped) = job.take_image() {
                        upload(ctx, &mut self.before, "image-before", &swapped.before);
                        upload(ctx, &mut self.after, "image-after", &swapped.after);
                        self.target_size = Some(swapped.result.dimensions());
                        self.result = Some(Arc::new(swapped.result));
                    }
                    if stats.faces == 0 && self.stats.is_none_or(|s| s.faces > 0) {
                        frame.actions.push(Action::Notify("No face found in this photo".into()));
                    }
                    self.stats = Some(stats);
                    self.failed = None;
                    self.job = None;
                }
                Status::Failed(error) => {
                    self.job = None;
                    self.fail(frame, error);
                }
                Status::Cancelled | Status::Done(_) => self.job = None,
                Status::Queued | Status::Loading | Status::Running(_) => {}
            }
        }
        match self.saving.poll() {
            Some(Ok(path)) => frame.actions.push(Action::Notify(format!("Saved {}", path.display()))),
            Some(Err(error)) => frame.actions.push(Action::Notify(error)),
            None => {}
        }
    }

    fn fail(&mut self, frame: &mut Frame<'_>, error: String) {
        frame.actions.push(Action::Notify(error.clone()));
        self.failed = Some(error);
    }

    /// Submit a swap once the face, photo or settings changed and stayed put for [`DEBOUNCE`].
    fn schedule(&mut self, ctx: &Context, frame: &Frame<'_>, runner: &JobRunner) {
        let (Some(source), Some(target)) = (&frame.snapshot.source_path, &self.target) else { return };
        if frame.source_pending {
            return; // a new face is on its way; wait for it
        }
        let request = Request { source: source.clone(), target: target.clone(), settings: frame.settings.clone() };
        if self.submitted.as_ref() == Some(&request) {
            self.wanted = None;
            return;
        }
        let now = ctx.input(|i| i.time);
        if self.wanted.as_ref().is_none_or(|(wanted, _)| *wanted != request) {
            self.wanted = Some((request.clone(), now));
        }
        let since = self.wanted.as_ref().map_or(now, |(_, since)| *since);
        if self.job.is_some() {
            return; // resubmitted when the running swap finishes
        }
        if now - since < DEBOUNCE {
            ctx.request_repaint_after(Duration::from_secs_f64(DEBOUNCE - (now - since)));
            return;
        }
        let Request { source, target, settings } = request.clone();
        self.job = Some(runner.submit(Job::Image { source, target, settings }));
        self.submitted = Some(request);
        self.failed = None;
    }

    fn empty(&mut self, ui: &mut Ui, rect: Rect) {
        let mut pick = false;
        let mut typed = None;
        let path_text = &mut self.path_text;
        centered(ui, column(rect, EMPTY_WIDTH), "image-empty", |ui| {
            let body = "Choose a photo with one or more faces. The source face and settings from the side panel are applied, \
                        and the result updates as you tweak them.";
            empty_state(ui, icons::IMAGE, "Swap faces in a photo", body, |ui| {
                if FilePicker::available() {
                    pick = ui.add(Button::filled("Choose photo").icon(icons::FOLDER_OPEN)).clicked();
                } else {
                    typed = path_field(ui, "Photo path", "Full path to a JPG or PNG", path_text);
                }
            });
        });
        if pick {
            self.open_picker(ui.ctx());
        }
        if let Some(path) = typed {
            self.set_target(ui.ctx(), path);
        }
    }

    fn open_picker(&self, ctx: &Context) {
        let ctx = ctx.clone();
        self.picker.open(Dialog::TargetImage, move || ctx.request_repaint());
    }

    /// File name and size on the left; change, save as and save on the right.
    fn toolbar(&mut self, ui: &mut Ui, rect: Rect) {
        let colors = theme::colors(ui.ctx());
        let target = self.target.clone().unwrap_or_default();
        let (mut change, mut save, mut save_as) = (false, false, false);
        let has_result = self.result.is_some();
        let saving = self.saving.is_running();
        ui.scope_builder(UiBuilder::new().max_rect(rect).layout(Layout::left_to_right(Align::Center)), |ui| {
            ui.spacing_mut().item_spacing.x = spacing::SM;
            let (icon_rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON), eframe::egui::Sense::hover());
            paint::icon(ui.painter(), icon_rect.center(), icons::IMAGE, size::ICON, colors.on_surface_variant, false);
            ui.add_space(spacing::XS);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                save = ui.add_enabled(has_result && !saving, Button::filled("Save").icon(icons::SAVE)).clicked();
                save_as = ui.add_enabled(has_result && !saving, Button::outlined("Save as…").icon(icons::SAVE_AS)).clicked();
                if FilePicker::available() {
                    change = ui.add(Button::text("Change photo").icon(icons::FOLDER_OPEN)).clicked();
                }
                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    let name = target.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                    ui.add(Label::new(TypeRole::TitleMedium.rich(name, colors.on_surface)).truncate());
                    let detail = match self.target_size {
                        Some((w, h)) => format!("{w} × {h}"),
                        None => "Opening…".to_owned(),
                    };
                    ui.label(TypeRole::BodySmall.rich(detail, colors.on_surface_variant));
                });
            });
        });
        if change {
            self.open_picker(ui.ctx());
        }
        if save {
            match self.default_save_path(&target) {
                Some(path) => self.save(ui.ctx(), path),
                None => save_as = true,
            }
        }
        if save_as {
            let default = self.save_path.clone().unwrap_or_else(|| sibling_output(&target, save_extension(&target)));
            let ctx = ui.ctx().clone();
            self.save_picker.open(Dialog::Save { title: "Save swapped photo", default }, move || ctx.request_repaint());
        }
    }

    /// Where Save writes without asking: the last save, or a fresh `-wessh` file beside the target.
    fn default_save_path(&self, target: &Path) -> Option<PathBuf> {
        if let Some(path) = &self.save_path {
            return Some(path.clone());
        }
        let path = sibling_output(target, save_extension(target));
        (!path.exists()).then_some(path) // never overwrite a file this session didn't write
    }

    fn save(&mut self, ctx: &Context, path: PathBuf) {
        let Some(result) = self.result.clone() else { return };
        self.save_path = Some(path.clone());
        self.saving.spawn(ctx, move || result.save(&path).map(|()| path).map_err(|e| format!("Could not save: {e}")));
    }

    /// The comparison, or the target with the reason there is no result yet.
    fn viewer(&mut self, ui: &mut Ui, area: Rect, frame: &mut Frame<'_>) {
        // The bottom row holds the stat chips, so tall photos never run under them.
        let rect = Rect::from_min_max(area.min, pos2(area.right(), area.bottom() - CHIP_ROW));
        let working = self.job.as_ref().map(JobHandle::status);
        match (&self.before, &self.after) {
            (Some(before), Some(after)) if self.result.is_some() => {
                Compare::new(before, after, &mut self.split).show(ui, rect);
            }
            (Some(before), _) => {
                let image = before.size_vec2();
                let fitted = Rect::from_center_size(rect.center(), image * (rect.size() / image).min_elem());
                eframe::egui::Image::new(before).corner_radius(theme::shape::LARGE).paint_at(ui, fitted);
                // Recede the unswapped photo behind the notice drawn over it.
                let scrim = theme::colors(ui.ctx()).scrim.gamma_multiply(SCRIM_OPACITY);
                ui.painter().rect_filled(fitted, theme::shape::LARGE, scrim);
            }
            _ => {}
        }
        let has_source = frame.snapshot.source_path.is_some();
        if !has_source && !frame.source_pending {
            Self::needs_source(ui, rect, frame);
        } else if let Some(error) = self.failed.clone().filter(|_| self.result.is_none()) {
            notice(ui, rect, "image-failed", |ui| empty_state(ui, icons::ERROR, "Couldn't swap this photo", &error, |_| {}));
        } else if self.result.is_none() {
            let loading = frame.snapshot.phase == Phase::Loading || matches!(working, Some(Status::Loading));
            let title = match (frame.source_pending, &working) {
                (true, _) => "Reading the source face…",
                (false, Some(Status::Queued)) => "Waiting for the video render…",
                _ => "Swapping faces…",
            };
            let body = if loading { "Loading models the first time can take a few seconds" } else { "This takes a moment" };
            notice(ui, rect, "image-working", |ui| {
                ui.add(CircularProgress::new(crate::ui::components::feedback::PROGRESS_SIZE));
                ui.add_space(spacing::LG);
                ui.label(TypeRole::TitleMedium.rich(title, theme::colors(ui.ctx()).on_surface));
                ui.label(TypeRole::BodyMedium.rich(body, theme::colors(ui.ctx()).on_surface_variant));
            });
        }
        if self.result.is_some() && (working.is_some() || self.wanted.is_some()) {
            let bar = Rect::from_min_size(rect.left_top(), vec2(rect.width(), size::STROKE));
            ui.scope_builder(UiBuilder::new().max_rect(bar), |ui| ui.add(LinearProgress::indeterminate()));
        }
        if let Some(stats) = self.stats.filter(|_| self.result.is_some()) {
            let chips = Rect::from_min_max(pos2(area.left(), rect.bottom()), area.max);
            ui.scope_builder(UiBuilder::new().max_rect(chips).layout(Layout::left_to_right(Align::Max)), |ui| stat_chips(ui, &stats));
        }
    }

    fn needs_source(ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>) {
        let pick = frame.intake.pick;
        let mut clicked = false;
        notice(ui, rect, "image-needs-source", |ui| {
            let body = "Choose whose face to put into this photo in the Source face panel.";
            empty_state(ui, icons::FRAME_PERSON, "Add a source face", body, |ui| {
                if pick {
                    clicked = ui.add(Button::filled("Choose face").icon(icons::ADD_PHOTO_ALTERNATE)).clicked();
                }
            });
        });
        if clicked {
            frame.actions.push(Action::PickSource);
        }
    }
}

/// A card centred over the viewer.
fn notice(ui: &mut Ui, rect: Rect, id: &str, add: impl FnOnce(&mut Ui)) {
    centered(ui, column(rect, NOTICE_WIDTH), id, |ui| {
        Card::elevated().padding(spacing::XL).show(ui, |ui| {
            ui.vertical_centered(add);
        });
    });
}

/// Keep the target's format where it can be written, else PNG.
fn save_extension(target: &Path) -> &'static str {
    if has_extension(target, &IMAGE_EXTENSIONS) { if has_extension(target, &["png"]) { "png" } else { "jpg" } } else { FALLBACK_EXTENSION }
}
