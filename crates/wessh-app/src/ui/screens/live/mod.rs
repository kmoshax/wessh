//! The Live screen: camera preview in the main pane, settings in a supporting side sheet.
//!
//! The sheet (source face and swap settings) is shared with the Image and Video screens
//! through [`LiveScreen::show`], so all three use the same face and settings.

mod preview;
pub mod sections;
mod sheet;

use crate::live::{Image, LiveSession, Phase, Snapshot};
use crate::ui::components::Snackbar;
use crate::ui::file_picker::{Dialog, FilePicker, Intake};
use crate::ui::task::Task;
use eframe::egui::{self, ColorImage, Context, Rect, TextureHandle, TextureOptions, Ui};
use sections::Section;
use std::path::PathBuf;
use wessh_core::Settings;
use wessh_media::{CameraConfig, Devices, LoopbackStatus, OutputTarget, list_devices};

/// Camera and output choices made in the UI (applied on the next Start).
pub struct LiveForm {
    pub camera: CameraConfig,
    /// The camera device is typed rather than picked from the discovered list.
    pub camera_custom: bool,
    pub output_enabled: bool,
    pub output_target: String,
    /// The output target is typed rather than picked from the discovered list.
    pub output_custom: bool,
    /// Typed source path, used only where neither a file dialog nor drag-and-drop exists.
    pub source_path: String,
}

impl Default for LiveForm {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            camera_custom: false,
            output_enabled: false,
            output_target: OutputTarget::default_for_platform().as_str().to_owned(),
            output_custom: false,
            source_path: String::new(),
        }
    }
}

impl LiveForm {
    /// After a scan, move defaults that don't exist on this machine to the first real device.
    fn adopt(&mut self, devices: &Devices) {
        if !self.camera_custom
            && !devices.cameras.iter().any(|d| d.id == self.camera.device)
            && let Some(first) = devices.cameras.first()
        {
            self.camera.device.clone_from(&first.id);
        }
        let is_default_output = self.output_target == OutputTarget::default_for_platform().as_str();
        if !self.output_custom
            && is_default_output
            && !devices.outputs.iter().any(|d| d.id == self.output_target)
            && let Some(first) = devices.outputs.first()
        {
            self.output_target.clone_from(&first.id);
        }
    }
}

/// Something the user asked for this frame, executed after drawing.
#[derive(Clone, Debug)]
pub enum Action {
    PickSource,
    LoadSource(PathBuf),
    ToggleLive,
    RefreshDevices,
    /// Show a snackbar message.
    Notify(String),
}

/// Everything a pane needs to draw one frame.
pub struct Frame<'a> {
    pub snapshot: &'a Snapshot,
    pub settings: &'a mut Settings,
    pub form: &'a mut LiveForm,
    pub preview: Option<&'a TextureHandle>,
    pub source: Option<&'a TextureHandle>,
    /// A source photo is being processed by the worker.
    pub source_pending: bool,
    pub intake: Intake,
    /// Cameras and virtual cameras found by the last scan.
    pub devices: Option<&'a Devices>,
    /// A device scan is running.
    pub scanning: bool,
    /// Drawn for the Image or Video screen: hide live-only options.
    pub offline: bool,
    pub actions: Vec<Action>,
}

impl Frame<'_> {
    /// Whether the camera is running.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.snapshot.phase == Phase::Live
    }
}

/// Seconds without feedback after which a source request is assumed finished
/// (the worker reports repeated identical errors without a visible change).
const SOURCE_TIMEOUT: f64 = 20.0;

/// A source photo sent to the worker but not yet accepted or rejected.
struct PendingSource {
    notice_before: Option<String>,
    since: f64,
}

/// State of the Live screen (and the shared settings sheet) that persists between frames.
#[derive(Default)]
pub struct LiveScreen {
    preview: Option<TextureHandle>,
    source: Option<TextureHandle>,
    form: LiveForm,
    snackbar: Snackbar,
    last_notice: Option<String>,
    picker: FilePicker,
    pending_source: Option<PendingSource>,
    devices: Option<Devices>,
    scan: Task<Devices>,
    /// See [`LiveScreen::pretend_loopback_missing`].
    loopback_missing: bool,
    /// Modal side sheet open (compact layouts only).
    sheet_open: bool,
}

impl LiveScreen {
    /// Pull new frames, thumbnails, notices and device lists.
    pub fn sync(&mut self, ctx: &Context, session: &LiveSession, snapshot: &Snapshot) {
        if let Some(image) = session.take_preview() {
            upload(ctx, &mut self.preview, "preview", &image);
            session.recycle_preview(image);
        }
        if let Some(image) = session.take_source_thumbnail() {
            upload(ctx, &mut self.source, "source", &image);
            self.pending_source = None;
        }
        self.settle_pending_source(ctx, snapshot);
        if snapshot.notice != self.last_notice {
            self.last_notice.clone_from(&snapshot.notice);
            if let Some(notice) = &snapshot.notice {
                self.notify(ctx, notice.clone());
            }
        }
        if let Some(path) = self.picker.take() {
            self.request_source(ctx, session, snapshot, path);
        }
        if self.devices.is_none() && !self.scan.is_running() {
            self.scan.spawn(ctx, list_devices);
        }
        if let Some(mut devices) = self.scan.poll() {
            if self.loopback_missing {
                devices.outputs.clear();
                devices.loopback = LoopbackStatus::NotLoaded;
            }
            self.form.adopt(&devices);
            self.devices = Some(devices);
        }
    }

    /// Design review: report v4l2loopback as not loaded whatever the scan finds.
    pub fn pretend_loopback_missing(&mut self) {
        self.loopback_missing = true;
    }

    /// Show a snackbar message.
    pub fn notify(&mut self, ctx: &Context, message: String) {
        self.snackbar.show(message, ctx.input(|i| i.time));
    }

    /// Send a source photo to the worker and show progress until it answers.
    pub fn request_source(&mut self, ctx: &Context, session: &LiveSession, snapshot: &Snapshot, path: PathBuf) {
        session.set_source(path);
        self.pending_source = Some(PendingSource { notice_before: snapshot.notice.clone(), since: ctx.input(|i| i.time) });
    }

    /// A new notice means the worker rejected the photo; give up after a timeout otherwise.
    fn settle_pending_source(&mut self, ctx: &Context, snapshot: &Snapshot) {
        let now = ctx.input(|i| i.time);
        let Some(pending) = &mut self.pending_source else { return };
        if snapshot.phase == Phase::Loading {
            pending.since = now; // the worker only reads commands once models are loaded
        }
        if snapshot.notice != pending.notice_before || now - pending.since > SOURCE_TIMEOUT {
            self.pending_source = None;
        } else {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(SOURCE_TIMEOUT));
        }
    }

    /// Toggle the modal settings sheet (compact layouts).
    pub fn toggle_sheet(&mut self) {
        self.sheet_open = !self.sheet_open;
    }

    /// Draw the Live screen into `ui`; `expanded` selects the two-pane layout.
    pub fn ui(&mut self, ui: &mut Ui, session: &LiveSession, snapshot: &Snapshot, expanded: bool) {
        self.show(ui, session, snapshot, expanded, false, preview::pane);
    }

    /// Draw `content` beside the settings sheet (docked when `expanded`, modal otherwise);
    /// `offline` drops the camera, output and live-only options for the Image and Video
    /// screens. `content` gets the rect left of the sheet and the shared frame.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        session: &LiveSession,
        snapshot: &Snapshot,
        expanded: bool,
        offline: bool,
        content: impl FnOnce(&mut Ui, Rect, &mut Frame<'_>),
    ) {
        let sections: &[Section] = if offline { &sections::OFFLINE } else { &sections::LIVE };
        let mut settings = session.settings();
        let mut frame = Frame {
            snapshot,
            settings: &mut settings,
            form: &mut self.form,
            preview: self.preview.as_ref(),
            source: self.source.as_ref(),
            source_pending: self.pending_source.is_some(),
            intake: Intake::detect(self.picker.is_open()),
            devices: self.devices.as_ref(),
            scanning: self.scan.is_running(),
            offline,
            actions: Vec::new(),
        };
        let area = ui.available_rect_before_wrap();
        let content_rect = if expanded {
            let (content_rect, sheet_rect) = sheet::split(area);
            sheet::docked(ui, sheet_rect, &mut frame, sections);
            content_rect
        } else {
            area
        };
        content(ui, content_rect, &mut frame);
        self.sheet_open &= !expanded;
        sheet::modal(ui.ctx(), &mut frame, &mut self.sheet_open, sections);

        let actions = std::mem::take(&mut frame.actions);
        if settings != session.settings() {
            session.set_settings(settings);
        }
        for action in actions {
            self.run(ui.ctx(), session, snapshot, action);
        }
        self.snackbar.ui(ui, content_rect);
    }

    /// Start or stop the camera.
    pub fn toggle_live(&self, session: &LiveSession, snapshot: &Snapshot) {
        match snapshot.phase {
            Phase::Live => session.stop(),
            Phase::Ready => {
                let output = self.form.output_enabled.then(|| OutputTarget::parse(self.form.output_target.trim()));
                session.start(self.form.camera.clone(), output);
            }
            Phase::Loading | Phase::Failed(_) => {}
        }
    }

    fn run(&mut self, ctx: &Context, session: &LiveSession, snapshot: &Snapshot, action: Action) {
        match action {
            Action::PickSource => {
                let ctx = ctx.clone();
                self.picker.open(Dialog::SourceImage, move || ctx.request_repaint());
            }
            Action::LoadSource(path) => self.request_source(ctx, session, snapshot, path),
            Action::ToggleLive => self.toggle_live(session, snapshot),
            Action::RefreshDevices => {
                if !self.scan.is_running() {
                    self.scan.spawn(ctx, list_devices);
                }
            }
            Action::Notify(message) => self.notify(ctx, message),
        }
    }
}

/// Upload `image` into `slot`, creating the texture on first use.
pub fn upload(ctx: &Context, slot: &mut Option<TextureHandle>, name: &str, image: &Image) {
    let pixels = ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba);
    match slot {
        Some(texture) => texture.set(pixels, TextureOptions::LINEAR),
        None => *slot = Some(ctx.load_texture(name, pixels, TextureOptions::LINEAR)),
    }
}

/// Paint `texture` so it covers `rect` (centre-cropped), with rounded corners.
pub fn paint_cover(ui: &Ui, texture: &TextureHandle, rect: Rect, radius: egui::CornerRadius) {
    let image = texture.size_vec2();
    let scale = (rect.size() / image).max_elem();
    let visible = rect.size() / (image * scale);
    let uv = Rect::from_center_size(egui::pos2(0.5, 0.5), visible);
    egui::Image::new(texture).uv(uv).corner_radius(radius).paint_at(ui, rect);
}
