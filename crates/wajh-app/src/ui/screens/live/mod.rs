//! The Live screen: camera preview in the main pane, settings in a supporting side sheet.

mod preview;
mod sections;
mod sheet;

use crate::live::{Image, LiveSession, Phase, Snapshot};
use crate::ui::components::Snackbar;
use crate::ui::file_picker::{FilePicker, Intake};
use eframe::egui::{self, ColorImage, Context, Rect, TextureHandle, TextureOptions, Ui};
use std::path::PathBuf;
use wajh_core::Settings;
use wajh_media::{CameraConfig, OutputTarget};

/// Camera and output choices made in the UI (applied on the next Start).
pub struct LiveForm {
    pub camera: CameraConfig,
    pub output_enabled: bool,
    pub output_target: String,
    /// Typed source path, used only where neither a file dialog nor drag-and-drop exists.
    pub source_path: String,
}

impl Default for LiveForm {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            output_enabled: false,
            output_target: OutputTarget::default_for_platform().as_str().to_owned(),
            source_path: String::new(),
        }
    }
}

/// Something the user asked for this frame, executed after drawing.
#[derive(Clone, Debug)]
pub enum Action {
    PickSource,
    LoadSource(PathBuf),
    ToggleLive,
}

/// Everything a pane needs to draw one frame.
pub struct Frame<'a> {
    pub snapshot: &'a Snapshot,
    pub settings: &'a mut Settings,
    pub form: &'a mut LiveForm,
    pub preview: Option<&'a TextureHandle>,
    pub source: Option<&'a TextureHandle>,
    pub intake: Intake,
    pub actions: Vec<Action>,
}

impl Frame<'_> {
    /// Whether the camera is running.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.snapshot.phase == Phase::Live
    }
}

/// State of the Live screen that persists between frames.
#[derive(Default)]
pub struct LiveScreen {
    preview: Option<TextureHandle>,
    source: Option<TextureHandle>,
    form: LiveForm,
    snackbar: Snackbar,
    last_notice: Option<String>,
    picker: FilePicker,
    /// Modal side sheet open (compact layouts only).
    sheet_open: bool,
}

impl LiveScreen {
    /// Pull new frames, thumbnails and notices from the session.
    pub fn sync(&mut self, ctx: &Context, session: &LiveSession, snapshot: &Snapshot) {
        if let Some(image) = session.take_preview() {
            upload(ctx, &mut self.preview, "preview", &image);
        }
        if let Some(image) = session.take_source_thumbnail() {
            upload(ctx, &mut self.source, "source", &image);
        }
        if snapshot.notice != self.last_notice {
            self.last_notice.clone_from(&snapshot.notice);
            if let Some(notice) = &snapshot.notice {
                self.snackbar.show(notice.clone(), ctx.input(|i| i.time));
            }
        }
        if let Some(path) = self.picker.take() {
            session.set_source(path);
        }
    }

    /// Toggle the modal settings sheet (compact layouts).
    pub fn toggle_sheet(&mut self) {
        self.sheet_open = !self.sheet_open;
    }

    /// Draw the screen into `ui`; `expanded` selects the two-pane layout.
    pub fn ui(&mut self, ui: &mut Ui, session: &LiveSession, snapshot: &Snapshot, expanded: bool) {
        let mut settings = session.settings();
        let mut frame = Frame {
            snapshot,
            settings: &mut settings,
            form: &mut self.form,
            preview: self.preview.as_ref(),
            source: self.source.as_ref(),
            intake: Intake::detect(self.picker.is_open()),
            actions: Vec::new(),
        };
        let content = ui.available_rect_before_wrap();
        let preview_rect = if expanded {
            let (preview_rect, sheet_rect) = sheet::split(content);
            sheet::docked(ui, sheet_rect, &mut frame);
            preview_rect
        } else {
            content
        };
        preview::pane(ui, preview_rect, &mut frame);
        self.sheet_open &= !expanded;
        sheet::modal(ui.ctx(), &mut frame, &mut self.sheet_open);
        self.snackbar.ui(ui, preview_rect);

        let actions = std::mem::take(&mut frame.actions);
        if settings != session.settings() {
            session.set_settings(settings);
        }
        for action in actions {
            self.run(ui.ctx(), session, snapshot, action);
        }
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

    fn run(&self, ctx: &Context, session: &LiveSession, snapshot: &Snapshot, action: Action) {
        match action {
            Action::PickSource => {
                let ctx = ctx.clone();
                self.picker.open(move || ctx.request_repaint());
            }
            Action::LoadSource(path) => session.set_source(path),
            Action::ToggleLive => self.toggle_live(session, snapshot),
        }
    }
}

fn upload(ctx: &Context, slot: &mut Option<TextureHandle>, name: &str, image: &Image) {
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
