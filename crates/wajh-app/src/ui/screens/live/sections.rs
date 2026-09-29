//! Settings side-sheet sections, one small function each.
//!
//! To add a section, write a `fn(&mut Ui, &mut Frame)` that wraps its content in
//! [`section`] (or [`toggle_section`]) and list it in [`ALL`].

use super::{Action, Frame, paint_cover};
use crate::ui::components::{Button, Card, Chip, Dropdown, ListItem, SegmentedButton, Slider, Switch, TextField, paint};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{Align, Align2, CursorIcon, Key, Label, Layout, Sense, Stroke, StrokeKind, TextureHandle, Ui, Vec2, pos2, vec2};
use std::path::PathBuf;
use wajh_core::assets::SwapModel;
use wajh_media::OutputTarget;

/// A settings section.
pub type Section = fn(&mut Ui, &mut Frame<'_>);

/// Sections in display order.
pub const ALL: [Section; 6] = [source, camera, output, swap, mask, enhance];

// ── Section chrome ─────────────────────────────────────────────────────────────

const HEADER_HEIGHT: f32 = 40.0;

/// A card with an icon + title header and `body` below.
fn section(ui: &mut Ui, icon: char, title: &str, body: impl FnOnce(&mut Ui)) {
    Card::elevated().show(ui, |ui| {
        header(ui, icon, title, |_| {});
        body(ui);
    });
}

/// A section whose header carries an on/off switch; the body is dimmed while off.
fn toggle_section(ui: &mut Ui, icon: char, title: &str, on: &mut bool, body: impl FnOnce(&mut Ui)) {
    Card::elevated().show(ui, |ui| {
        header(ui, icon, title, |ui| {
            ui.add(Switch::new(on));
        });
        let enabled = *on;
        ui.add_enabled_ui(enabled, body);
    });
}

fn header(ui: &mut Ui, icon: char, title: &str, trailing: impl FnOnce(&mut Ui)) {
    let colors = theme::colors(ui.ctx());
    ui.allocate_ui_with_layout(vec2(ui.available_width(), HEADER_HEIGHT), Layout::left_to_right(Align::Center), |ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON), Sense::hover());
        paint::icon(ui.painter(), rect.center(), icon, size::ICON, colors.primary, false);
        ui.add_space(spacing::XS);
        ui.label(TypeRole::TitleMedium.rich(title, colors.on_surface));
        ui.with_layout(Layout::right_to_left(Align::Center), trailing);
    });
}

/// A small caption above a control group.
fn caption(ui: &mut Ui, text: &str) {
    let colors = theme::colors(ui.ctx());
    ui.add_space(spacing::XS);
    ui.label(TypeRole::LabelLarge.rich(text, colors.on_surface_variant));
}

/// A caption with the formatted value on the right, followed by a slider.
fn labeled_slider(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, format: fn(f32) -> String) {
    let colors = theme::colors(ui.ctx());
    let enabled = ui.is_enabled();
    ui.horizontal(|ui| {
        ui.label(TypeRole::BodyLarge.rich(label, paint::content(colors.on_surface, enabled)));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(TypeRole::LabelLarge.rich(format(*value), paint::content(colors.on_surface_variant, enabled)));
        });
    });
    ui.add(Slider::new(value, range).format(format));
}

/// A list row with a trailing switch; clicking anywhere on the row toggles it.
fn switch_row(ui: &mut Ui, icon: char, headline: &str, supporting: &str, on: &mut bool) {
    let (row, ()) = ListItem::new(headline).supporting(supporting).icon(icon).show(ui, |ui| {
        ui.add(Switch::new(on));
    });
    if row.clicked() {
        *on = !*on;
    }
}

fn percent(value: f32) -> String {
    format!("{:.0} %", value * 100.0)
}

fn percent_points(value: f32) -> String {
    format!("{value:.0} %")
}

// ── Source face ────────────────────────────────────────────────────────────────

const THUMBNAIL: f32 = 72.0;
const DROP_ZONE_HEIGHT: f32 = 184.0;
const DROP_ICON_CIRCLE: f32 = 56.0;

fn source(ui: &mut Ui, frame: &mut Frame<'_>) {
    section(ui, icons::FACE, "Source face", |ui| {
        match frame.source {
            Some(texture) if frame.snapshot.has_source => source_loaded(ui, frame, texture),
            _ => drop_zone(ui, frame),
        }
        if frame.intake.needs_path_field() {
            path_field(ui, frame);
        }
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

fn camera(ui: &mut Ui, frame: &mut Frame<'_>) {
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

fn output(ui: &mut Ui, frame: &mut Frame<'_>) {
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

// ── Swap ───────────────────────────────────────────────────────────────────────

const MODELS: [(SwapModel, &str); 4] = [
    (SwapModel::Inswapper, SwapModel::Inswapper.label()),
    (SwapModel::Hyperswap1a, SwapModel::Hyperswap1a.label()),
    (SwapModel::Hyperswap1b, SwapModel::Hyperswap1b.label()),
    (SwapModel::Hyperswap1c, SwapModel::Hyperswap1c.label()),
];
const DETECTOR_SIZES: [(usize, &str); 3] = [(320, "320"), (480, "480"), (640, "640")];

fn swap(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    section(ui, icons::SWAP_HORIZ, "Swap", |ui| {
        ui.add(Dropdown::new("Model", &mut settings.swap_model, &MODELS).icon(icons::MEMORY));
        switch_row(ui, icons::GROUPS, "All faces", "Swap every face, not just the largest", &mut settings.many_faces);
        switch_row(ui, icons::FLIP, "Mirror", "Flip the picture like a selfie camera", &mut settings.mirror);
        switch_row(ui, icons::CENTER_FOCUS_STRONG, "Stabilise", "Smooth landmark jitter between frames", &mut settings.stabilize);
        caption(ui, "Detector size");
        ui.add(SegmentedButton::new(&mut settings.detector_size, &DETECTOR_SIZES));
    });
}

// ── Mask ───────────────────────────────────────────────────────────────────────

const MAX_BLUR: f32 = 1.0;
const MAX_PADDING: f32 = 50.0;
const PADDING_SIDES: [&str; 4] = ["Top", "Right", "Bottom", "Left"];

fn mask(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    section(ui, icons::BLUR_ON, "Mask", |ui| {
        labeled_slider(ui, "Edge blur", &mut settings.mask_blur, 0.0..=MAX_BLUR, percent);
        caption(ui, "Padding");
        ui.spacing_mut().item_spacing.x = spacing::XL;
        let [top, right, bottom, left] = &mut settings.mask_padding;
        for row in [[(PADDING_SIDES[0], top), (PADDING_SIDES[1], right)], [(PADDING_SIDES[2], bottom), (PADDING_SIDES[3], left)]] {
            ui.columns(2, |columns| {
                for (column, (side, value)) in columns.iter_mut().zip(row) {
                    labeled_slider(column, side, value, 0.0..=MAX_PADDING, percent_points);
                }
            });
        }
    });
}

// ── Enhance ────────────────────────────────────────────────────────────────────

fn enhance(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    toggle_section(ui, icons::AUTO_AWESOME, "Enhance", &mut settings.enhance, |ui| {
        let colors = theme::colors(ui.ctx());
        let enabled = ui.is_enabled();
        let job = TypeRole::BodyMedium.job(
            "Restore skin and eye detail with GPEN. Costs a few milliseconds per face.",
            paint::content(colors.on_surface_variant, enabled),
            Some(ui.available_width()),
        );
        let galley = ui.fonts_mut(|f| f.layout_job(job));
        ui.add(eframe::egui::Label::new(galley).selectable(false));
        labeled_slider(ui, "Blend", &mut settings.enhance_blend, 0.0..=1.0, percent);
    });
}
