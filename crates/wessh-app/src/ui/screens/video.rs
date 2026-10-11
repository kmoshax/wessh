//! The Video screen: pick a clip, see its details, choose output and encoder, and render it
//! with the shared source face and settings on the job worker, with progress and a live frame.

use super::layout::{centered, column, pane, path_field};
use super::live::{Action, Frame, upload};
use crate::jobs::image::{THUMBNAIL_MAX, thumbnail};
use crate::jobs::video::Progress;
use crate::jobs::{Finished, Job, JobHandle, JobRunner, Status};
use crate::live::Image;
use crate::ui::components::{Button, Chip, Dropdown, LinearProgress, TextField, empty_state, paint};
use crate::ui::file_picker::{Dialog, FilePicker, reveal, sibling_output};
use crate::ui::icons;
use crate::ui::task::Task;
use crate::ui::theme::{self, TypeRole, shape, size, spacing};
use crate::ui::tools::VideoTools;
use eframe::egui::{Align, Context, Id, Label, Layout, Rect, Sense, TextureHandle, Ui, UiBuilder, Vec2, pos2, vec2};
use std::path::{Path, PathBuf};
use wessh_media::{Encoder, VideoInfo, VideoReader, probe};

/// Inset of the media and controls from the pane edges.
const INSET: f32 = spacing::LG;
/// Controls height assumed before the first measurement.
const CONTROLS_GUESS: f32 = 160.0;
const EMPTY_WIDTH: f32 = 420.0;
/// Width of the encoder dropdown.
const ENCODER_WIDTH: f32 = 300.0;
const OUTPUT_EXTENSION: &str = "mp4";
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3600;

/// What the probe thread found out about the target.
struct Probed {
    info: VideoInfo,
    poster: Option<Image>,
}

/// State of the Video screen.
#[derive(Default)]
pub struct VideoScreen {
    target: Option<PathBuf>,
    path_text: String,
    output: String,
    picker: FilePicker,
    save_picker: FilePicker,
    probing: Task<Result<Probed, String>>,
    info: Option<VideoInfo>,
    /// First frame of the target, shown until rendering starts.
    poster: Option<TextureHandle>,
    /// Newest rendered frame.
    frame: Option<TextureHandle>,
    listing: Task<Vec<Encoder>>,
    encoders: Option<Vec<Encoder>>,
    encoder: usize,
    job: Option<JobHandle>,
    /// Last status of `job`, kept after it ends.
    status: Option<Status>,
    probe_error: Option<String>,
    /// Start rendering as soon as everything is ready (screenshot capture).
    autostart: bool,
    /// The ffmpeg that decoding and encoding need, downloaded on first use.
    tools: VideoTools,
}

impl VideoScreen {
    /// Render `path` from now on.
    pub fn set_target(&mut self, ctx: &Context, path: PathBuf) {
        if self.is_running() {
            return;
        }
        let load = path.clone();
        self.probing.spawn(ctx, move || probe_video(&load));
        self.output = sibling_output(&path, OUTPUT_EXTENSION).to_string_lossy().into_owned();
        self.target = Some(path);
        (self.info, self.poster, self.frame, self.status, self.probe_error) = (None, None, None, None, None);
    }

    /// Write to `path` instead of the default.
    pub fn set_output(&mut self, path: &Path) {
        self.output = path.to_string_lossy().into_owned();
    }

    /// Start rendering once the target, face and encoders are ready.
    pub fn render_when_ready(&mut self) {
        self.autostart = true;
    }

    fn is_running(&self) -> bool {
        self.job.is_some()
    }

    /// Draw the screen into `rect`; jobs go to `runner`.
    pub fn ui(&mut self, ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>, runner: &JobRunner) {
        let ctx = ui.ctx().clone();
        if !self.tools.is_ready() {
            pane(ui, rect);
            centered(ui, column(rect, EMPTY_WIDTH), "video-tools", |ui| self.tools.ui(ui, "Videos"));
            return;
        }
        self.poll(&ctx, frame);
        if self.autostart && self.ready(frame) {
            self.autostart = false;
            self.start(frame, runner);
        }
        pane(ui, rect);
        if self.target.is_none() {
            self.empty(ui, rect);
            return;
        }
        let inner = rect.shrink(INSET);
        let id = Id::new("video-controls-height");
        let controls_height: f32 = ui.ctx().data(|d| d.get_temp(id)).unwrap_or(CONTROLS_GUESS);
        let controls = Rect::from_min_max(pos2(inner.left(), inner.bottom() - controls_height), inner.max);
        let media = Rect::from_min_max(inner.min, pos2(inner.right(), controls.top() - spacing::LG));
        self.media(ui, media);
        let used = ui
            .scope_builder(UiBuilder::new().max_rect(controls).layout(Layout::top_down(Align::Min)), |ui| self.controls(ui, frame, runner));
        let height = used.response.rect.height();
        if (height - controls_height).abs() > 0.5 {
            ui.ctx().data_mut(|d| d.insert_temp(id, height));
            ui.ctx().request_repaint();
        }
    }

    fn poll(&mut self, ctx: &Context, frame: &mut Frame<'_>) {
        if let Some(path) = self.picker.take() {
            self.set_target(ctx, path);
        }
        if let Some(path) = self.save_picker.take() {
            self.set_output(&path);
        }
        if self.encoders.is_none() && !self.listing.is_running() {
            self.listing.spawn(ctx, Encoder::available);
        }
        if let Some(mut encoders) = self.listing.poll() {
            if encoders.is_empty() {
                encoders.push(Encoder::best_available());
            }
            self.encoders = Some(encoders);
        }
        match self.probing.poll() {
            Some(Ok(probed)) => {
                if let Some(poster) = &probed.poster {
                    upload(ctx, &mut self.poster, "video-poster", poster);
                }
                self.info = Some(probed.info);
            }
            Some(Err(error)) => {
                frame.actions.push(Action::Notify(error.clone()));
                self.probe_error = Some(error);
            }
            None => {}
        }
        if let Some(job) = &self.job {
            if let Some(image) = job.take_preview() {
                upload(ctx, &mut self.frame, "video-frame", &image);
            }
            let status = job.status();
            if status.is_final() {
                match &status {
                    Status::Done(Finished::Video { output, .. }) => {
                        let name = output.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                        frame.actions.push(Action::Notify(format!("Rendered {name}")));
                    }
                    Status::Failed(error) => frame.actions.push(Action::Notify(error.clone())),
                    _ => {}
                }
                self.job = None;
            }
            self.status = Some(status);
        }
    }

    /// Target probed, encoders listed and a source face loaded.
    fn ready(&self, frame: &Frame<'_>) -> bool {
        self.info.is_some()
            && self.encoders.is_some()
            && frame.snapshot.source_path.is_some()
            && !frame.source_pending
            && !self.is_running()
    }

    fn start(&mut self, frame: &mut Frame<'_>, runner: &JobRunner) {
        let (Some(target), Some(info), Some(source)) = (&self.target, &self.info, &frame.snapshot.source_path) else { return };
        let output = PathBuf::from(self.output.trim());
        if output.as_os_str().is_empty() || same_file(&output, target) {
            frame.actions.push(Action::Notify("Choose an output file other than the video itself".into()));
            return;
        }
        let encoder = self.encoders.as_ref().and_then(|e| e.get(self.encoder).cloned()).unwrap_or_else(Encoder::best_available);
        let job = Job::Video {
            source: source.clone(),
            target: target.clone(),
            output,
            encoder,
            info: info.clone(),
            settings: frame.settings.clone(),
        };
        self.job = Some(runner.submit(job));
        self.status = Some(Status::Queued);
        self.frame = None;
    }

    fn empty(&mut self, ui: &mut Ui, rect: Rect) {
        let (mut pick, mut typed) = (false, None);
        let path_text = &mut self.path_text;
        centered(ui, column(rect, EMPTY_WIDTH), "video-empty", |ui| {
            let body = "Every frame gets the source face and settings from the side panel. The original audio is kept, \
                        and encoding runs on the GPU when it can.";
            empty_state(ui, icons::MOVIE, "Swap faces in a video", body, |ui| {
                if FilePicker::available() {
                    pick = ui.add(Button::filled("Choose video").icon(icons::VIDEO_FILE)).clicked();
                } else {
                    typed = path_field(ui, "Video path", "Full path to an MP4, MOV, MKV…", path_text);
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
        self.picker.open(Dialog::TargetVideo, move || ctx.request_repaint());
    }

    /// The newest rendered frame (or the poster), with the clip's details over it.
    fn media(&self, ui: &mut Ui, rect: Rect) {
        let colors = theme::colors(ui.ctx());
        let mut shown = rect;
        if let Some(texture) = self.frame.as_ref().or(self.poster.as_ref()) {
            let image = texture.size_vec2();
            shown = Rect::from_center_size(rect.center(), image * (rect.size() / image).min_elem());
            eframe::egui::Image::new(texture).corner_radius(shape::LARGE).paint_at(ui, shown);
        } else if let Some(error) = &self.probe_error {
            centered(ui, column(rect, EMPTY_WIDTH), "video-probe-failed", |ui| {
                empty_state(ui, icons::ERROR, "Couldn't read this video", error, |_| {});
            });
        } else {
            ui.painter().rect_filled(rect, shape::LARGE, colors.surface_container);
            centered(ui, rect, "video-probing", |ui| {
                ui.add(crate::ui::components::CircularProgress::new(crate::ui::components::feedback::PROGRESS_SIZE));
            });
        }
        let Some(info) = &self.info else { return };
        let chips = Rect::from_min_size(shown.min + Vec2::splat(spacing::MD), vec2(shown.width() - 2.0 * spacing::MD, size::TARGET));
        ui.scope_builder(UiBuilder::new().max_rect(chips).layout(Layout::left_to_right(Align::Min)), |ui| {
            ui.spacing_mut().item_spacing.x = spacing::SM;
            ui.add(Chip::status(&format!("{} × {}", info.width, info.height), false).icon(icons::ASPECT_RATIO));
            ui.add(Chip::status(&format!("{:.3} fps", info.fps).replace(".000", ""), false).icon(icons::SPEED));
            if let Some(seconds) = info.seconds() {
                ui.add(Chip::status(&clock(seconds), false).icon(icons::SCHEDULE));
            }
            if let Some(frames) = info.total_frames() {
                ui.add(Chip::status(&format!("{frames} frames"), false).icon(icons::BURST_MODE));
            }
        });
    }

    fn controls(&mut self, ui: &mut Ui, frame: &mut Frame<'_>, runner: &JobRunner) {
        match self.status.clone() {
            Some(Status::Queued | Status::Loading | Status::Running(_)) if self.is_running() => self.progress(ui),
            Some(Status::Done(Finished::Video { output, progress })) => self.done(ui, &output, &progress),
            status => self.setup(ui, frame, runner, status.as_ref()),
        }
    }

    /// Output, encoder and the Render button.
    fn setup(&mut self, ui: &mut Ui, frame: &mut Frame<'_>, runner: &JobRunner, last: Option<&Status>) {
        let colors = theme::colors(ui.ctx());
        ui.spacing_mut().item_spacing = vec2(spacing::MD, spacing::MD);
        let browse = FilePicker::available();
        let (mut open_browse, mut change, mut render) = (false, false, false);
        ui.horizontal(|ui| {
            let button_room = if browse { paint::text_width(ui, TypeRole::LabelLarge, "Browse…") + 3.0 * spacing::XL } else { 0.0 };
            ui.allocate_ui(vec2(ui.available_width() - button_room, 0.0), |ui| {
                ui.add(TextField::new("Output file", &mut self.output).icon(icons::SAVE).hint("Where to write the result"));
            });
            if browse {
                open_browse = ui.add(Button::outlined("Browse…").icon(icons::FOLDER_OPEN)).clicked();
            }
        });
        let ready = self.ready(frame);
        ui.horizontal(|ui| {
            ui.allocate_ui(vec2(ENCODER_WIDTH.min(ui.available_width() / 2.0), 0.0), |ui| {
                if let Some(encoders) = &self.encoders {
                    let options: Vec<(usize, &str)> = encoders.iter().enumerate().map(|(i, e)| (i, e.label())).collect();
                    ui.add(Dropdown::new("Encoder", &mut self.encoder, &options).icon(icons::MEMORY));
                } else {
                    let mut detecting = 0;
                    ui.add_enabled(false, Dropdown::new("Encoder", &mut detecting, &[(0, "Detecting…")]).icon(icons::MEMORY));
                }
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                let label = if matches!(last, Some(Status::Cancelled | Status::Failed(_))) { "Render again" } else { "Render" };
                render = ui.add_enabled(ready, Button::filled(label).icon(icons::PLAY_ARROW)).clicked();
                if browse {
                    change = ui.add(Button::text("Change video").icon(icons::VIDEO_FILE)).clicked();
                }
            });
        });
        let hint = if frame.snapshot.source_path.is_none() {
            Some(("Add a source face in the side panel to render".to_owned(), colors.on_surface_variant))
        } else {
            match last {
                Some(Status::Failed(error)) => Some((format!("Render failed: {error}"), colors.error)),
                Some(Status::Cancelled) => Some(("Render cancelled — the partial file was removed".to_owned(), colors.on_surface_variant)),
                _ => None,
            }
        };
        if let Some((text, color)) = hint {
            ui.add(Label::new(TypeRole::BodyMedium.rich(text, color)).wrap());
        }
        if open_browse {
            let ctx = ui.ctx().clone();
            let default = PathBuf::from(self.output.trim());
            self.save_picker.open(Dialog::Save { title: "Save rendered video", default }, move || ctx.request_repaint());
        }
        if change {
            self.open_picker(ui.ctx());
        }
        if render {
            self.start(frame, runner);
        }
    }

    /// Progress bar, counts, speed, time left and Cancel.
    fn progress(&mut self, ui: &mut Ui) {
        let colors = theme::colors(ui.ctx());
        let progress = match &self.status {
            Some(Status::Running(progress)) => Some(*progress),
            _ => None,
        };
        let (title, detail) = match (&self.status, progress) {
            (_, Some(p)) => ("Rendering…", progress_line(&p)),
            (Some(Status::Queued), _) => ("Waiting…", "Another job is finishing first".to_owned()),
            _ => ("Preparing…", "Loading models and the source face".to_owned()),
        };
        let mut cancel = false;
        ui.spacing_mut().item_spacing.y = spacing::MD;
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = spacing::XXS;
                ui.label(TypeRole::TitleMedium.rich(title, colors.on_surface));
                ui.label(TypeRole::BodyMedium.rich(detail, colors.on_surface_variant));
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if let Some(eta) = progress.and_then(|p| p.eta()) {
                    cancel = ui.add(Button::outlined("Cancel").icon(icons::CANCEL)).clicked();
                    ui.add_space(spacing::LG);
                    ui.label(TypeRole::TitleMedium.rich(format!("{} left", clock(eta.as_secs_f64())), colors.on_surface));
                } else {
                    cancel = ui.add(Button::outlined("Cancel").icon(icons::CANCEL)).clicked();
                }
            });
        });
        match progress.and_then(|p| p.fraction()) {
            Some(fraction) => ui.add(LinearProgress::determinate(fraction)),
            None => ui.add(LinearProgress::indeterminate()),
        };
        if cancel && let Some(job) = &self.job {
            job.cancel();
        }
    }

    /// Where the result went, how long it took, and what next.
    fn done(&mut self, ui: &mut Ui, output: &Path, progress: &Progress) {
        let colors = theme::colors(ui.ctx());
        let (mut open, mut again) = (false, false);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = spacing::MD;
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON_LARGE), Sense::hover());
            paint::icon(ui.painter(), rect.center(), icons::TASK_ALT, size::ICON_LARGE, colors.primary, false);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                open = ui.add(Button::tonal("Open folder").icon(icons::FOLDER)).clicked();
                again = ui.add(Button::text("Render again").icon(icons::REPLAY)).clicked();
                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    ui.spacing_mut().item_spacing.y = spacing::XXS;
                    let name = output.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                    ui.add(Label::new(TypeRole::TitleMedium.rich(format!("Saved {name}"), colors.on_surface)).truncate());
                    let summary = format!(
                        "{} frames in {} · {:.1} fps average",
                        progress.done,
                        clock(progress.elapsed.as_secs_f64()),
                        progress.fps()
                    );
                    ui.label(TypeRole::BodyMedium.rich(summary, colors.on_surface_variant));
                    let folder = output.parent().map_or_else(String::new, |p| p.display().to_string());
                    ui.add(Label::new(TypeRole::BodySmall.rich(folder, colors.on_surface_variant)).truncate());
                });
            });
        });
        if open {
            reveal(output);
        }
        if again {
            self.status = None;
        }
    }
}

/// `probe` plus the first frame as a poster, off the UI thread.
fn probe_video(path: &Path) -> Result<Probed, String> {
    let name = path.to_string_lossy();
    let info = probe(&name).map_err(|e| format!("Could not read {}: {e}", path.display()))?;
    let poster =
        VideoReader::open(&name, &info).ok().and_then(|mut reader| reader.next()).map(|frame| thumbnail(&frame, THUMBNAIL_MAX * 2));
    Ok(Probed { info, poster })
}

fn same_file(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// "120 / 480 frames · 45.2 fps" (or without the total when unknown).
fn progress_line(progress: &Progress) -> String {
    match progress.total {
        Some(total) => format!("{} / {total} frames · {:.1} fps", progress.done, progress.fps()),
        None => format!("{} frames · {:.1} fps", progress.done, progress.fps()),
    }
}

/// `m:ss`, or `h:mm:ss` from an hour.
fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (hours, minutes, secs) = (total / SECONDS_PER_HOUR, total % SECONDS_PER_HOUR / SECONDS_PER_MINUTE, total % SECONDS_PER_MINUTE);
    if hours > 0 { format!("{hours}:{minutes:02}:{secs:02}") } else { format!("{minutes}:{secs:02}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn formats_clock() {
        assert_eq!(clock(0.4), "0:00");
        assert_eq!(clock(75.0), "1:15");
        assert_eq!(clock(3725.0), "1:02:05");
    }

    #[test]
    fn describes_progress() {
        let progress = Progress { done: 10, total: Some(40), elapsed: Duration::from_secs(2) };
        assert_eq!(progress_line(&progress), "10 / 40 frames · 5.0 fps");
        assert_eq!(progress_line(&Progress { total: None, ..progress }), "10 frames · 5.0 fps");
    }
}
