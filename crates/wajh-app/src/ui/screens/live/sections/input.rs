//! Where frames and the source face come from, and where results go.

use super::super::{Action, Frame, paint_cover};
use super::{caption, section, switch_row};
use crate::ui::components::{Button, Chip, CircularProgress, SegmentedButton, TextField, paint};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{Align2, CursorIcon, Key, Label, Sense, Stroke, StrokeKind, TextureHandle, Ui, Vec2, pos2, vec2};
use std::path::PathBuf;
use wajh_media::OutputTarget;

// ── Source face ────────────────────────────────────────────────────────────────

const THUMBNAIL: f32 = 72.0;
const PENDING_PROGRESS: f32 = 40.0;
const DROP_ZONE_HEIGHT: f32 = 184.0;
const DROP_ICON_CIRCLE: f32 = 56.0;

pub(super) fn source(ui: &mut Ui, frame: &mut Frame<'_>) {
    section(ui, icons::FACE, "Source face", |ui| {
        match frame.source {
            _ if frame.source_pending => source_pending(ui),
            Some(texture) if frame.snapshot.has_source => source_loaded(ui, frame, texture),
            _ => drop_zone(ui, frame),
        }
        if frame.intake.needs_path_field() {
            path_field(ui, frame);
        }
    });
}

/// Spinner shown between choosing a photo and the worker accepting (or rejecting) it.
fn source_pending(ui: &mut Ui) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing::LG;
        ui.add(CircularProgress::new(PENDING_PROGRESS));
        ui.vertical(|ui| {
            ui.label(TypeRole::TitleMedium.rich("Finding the face…", colors.on_surface));
            ui.label(TypeRole::BodyMedium.rich("Detecting and encoding the source photo", colors.on_surface_variant));
        });
    });
}

fn source_loaded(ui: &mut Ui, frame: &mut Frame<'_>, texture: &TextureHandle) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(THUMBNAIL), Sense::hover());
        paint_cover(ui, texture, rect, shape::MEDIUM);
        ui.add_space(spacing::SM);
        ui.vertical(|ui| {
            ui.add_space(spacing::XS);
            let name = frame.snapshot.source_name.as_deref().unwrap_or("Source face");
            ui.add(Label::new(TypeRole::TitleMedium.rich(name, colors.on_surface)).truncate());
            face_ready(ui);
            ui.add_space(spacing::XS);
            if frame.intake.pick && ui.add(Button::tonal("Replace").icon(icons::SWAP_HORIZ)).clicked() {
                frame.actions.push(Action::PickSource);
            }
        });
    });
    if frame.intake.drop && !frame.intake.pick {
        caption(ui, "Drop another photo anywhere to replace it");
    }
}

/// Filled check icon + "Face ready".
fn face_ready(ui: &mut Ui) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing::XS;
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON_SMALL), Sense::hover());
        paint::icon(ui.painter(), rect.center(), icons::CHECK_CIRCLE, size::ICON_SMALL, colors.primary, true);
        ui.label(TypeRole::BodyMedium.rich("Face ready", colors.on_surface_variant));
    });
}

/// Empty state: a large outlined target inviting a drop or a click.
fn drop_zone(ui: &mut Ui, frame: &mut Frame<'_>) {
    let colors = theme::colors(ui.ctx());
    let intake = frame.intake;
    let sense = if intake.pick { Sense::click() } else { Sense::hover() };
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), DROP_ZONE_HEIGHT), sense);
    let response = if intake.pick { response.on_hover_cursor(CursorIcon::PointingHand) } else { response };
    let hovered = ui.ctx().animate_bool_with_time(response.id, response.hovered() && intake.pick, motion::SHORT_4);
    let painter = ui.painter();
    let outline = colors.outline_variant.lerp_to_gamma(colors.primary, hovered);
    painter.rect_stroke(rect, shape::MEDIUM, Stroke::new(size::STROKE, outline), StrokeKind::Inside);
    paint::state_layer(ui, &response, rect, shape::MEDIUM, colors.primary);

    let circle_center = pos2(rect.center().x, rect.top() + spacing::XL + DROP_ICON_CIRCLE / 2.0);
    painter.circle_filled(circle_center, DROP_ICON_CIRCLE / 2.0, colors.primary_container);
    paint::icon(painter, circle_center, icons::ADD_PHOTO_ALTERNATE, size::ICON, colors.on_primary_container, false);
    let title_pos = pos2(rect.center().x, circle_center.y + DROP_ICON_CIRCLE / 2.0 + spacing::MD);
    let title = paint::text(ui, title_pos, Align2::CENTER_TOP, TypeRole::TitleMedium, intake.title(), colors.on_surface);
    let hint_pos = pos2(rect.center().x, title.bottom() + spacing::XS);
    paint::text(ui, hint_pos, Align2::CENTER_TOP, TypeRole::BodyMedium, intake.hint(), colors.on_surface_variant);
    if response.clicked() {
        frame.actions.push(Action::PickSource);
    }
}

/// Fallback when there is no file dialog and no drag-and-drop: type a path.
fn path_field(ui: &mut Ui, frame: &mut Frame<'_>) {
    let path = &mut frame.form.source_path;
    let submitted = ui.add(TextField::new("Image path", path).icon(icons::FOLDER_OPEN).hint("Full path to a JPG or PNG")).lost_focus()
        && ui.input(|i| i.key_pressed(Key::Enter));
    let clicked = ui.add_enabled(!path.trim().is_empty(), Button::tonal("Load").icon(icons::UPLOAD)).clicked();
    if submitted || clicked {
        frame.actions.push(Action::LoadSource(PathBuf::from(path.trim())));
    }
}

// ── Camera ─────────────────────────────────────────────────────────────────────

const RESOLUTIONS: [((usize, usize), &str); 3] = [((640, 480), "480p"), ((1280, 720), "720p"), ((1920, 1080), "1080p")];
const FRAME_RATES: [(u32, &str); 3] = [(15, "15"), (30, "30"), (60, "60")];

pub(super) fn camera(ui: &mut Ui, frame: &mut Frame<'_>) {
    let live = frame.is_live();
    let camera = &mut frame.form.camera;
    section(ui, icons::VIDEOCAM, "Camera", |ui| {
        ui.add_enabled_ui(!live, |ui| {
            ui.add(TextField::new("Device", &mut camera.device).icon(icons::PHOTO_CAMERA));
            caption(ui, "Resolution");
            let mut resolution = (camera.width, camera.height);
            if ui.add(SegmentedButton::new(&mut resolution, &RESOLUTIONS)).changed() {
                (camera.width, camera.height) = resolution;
            }
            caption(ui, "Frame rate");
            ui.add(SegmentedButton::new(&mut camera.fps, &FRAME_RATES));
            if live {
                caption(ui, "Stop the camera to change these");
            }
        });
    });
}

// ── Output ─────────────────────────────────────────────────────────────────────

pub(super) fn output(ui: &mut Ui, frame: &mut Frame<'_>) {
    let live = frame.is_live();
    let streaming = live && frame.snapshot.output_active;
    let form = &mut *frame.form;
    section(ui, icons::CAST, "Output", |ui| {
        ui.add_enabled_ui(!live, |ui| {
            let kind = match OutputTarget::parse(form.output_target.trim()) {
                OutputTarget::V4l2(_) => "Appear as a webcam in other apps",
                OutputTarget::Stream(_) => "Stream to OBS as a media source",
            };
            switch_row(ui, icons::SENSORS, "Virtual camera", kind, &mut form.output_enabled);
            ui.add_enabled_ui(form.output_enabled, |ui| {
                ui.add(TextField::new("Target", &mut form.output_target).hint("/dev/video10 or udp://…"));
            });
        });
        if streaming {
            ui.add(Chip::status("Sending frames", true).icon(icons::GRAPHIC_EQ));
        }
    });
}
