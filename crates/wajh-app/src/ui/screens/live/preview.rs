//! The preview pane: live video with status overlays, or the matching empty/loading/error state.

use super::{Action, Frame};
use crate::live::Phase;
use crate::ui::components::{Button, Chip, CircularProgress, ExtendedFab, FabColor, empty_state, feedback::PROGRESS_SIZE, paint};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, shape, spacing};
use eframe::egui::{Align, Align2, Id, Layout, Rect, TextureHandle, Ui, UiBuilder, Vec2, pos2, vec2};

/// Inset of overlay chips and the FAB from the pane edges.
const OVERLAY_INSET: f32 = spacing::LG;
const LIVE_BADGE_HEIGHT: f32 = 24.0;
const LIVE_DOT: f32 = 8.0;
/// Seconds per pulse of the LIVE dot.
const LIVE_PULSE_PERIOD: f64 = 1.6;
const LIVE_PULSE_MIN_OPACITY: f32 = 0.35;

/// Draw the preview pane into `rect`.
pub fn pane(ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>) {
    let colors = theme::colors(ui.ctx());
    ui.painter().rect_filled(rect, shape::EXTRA_LARGE, colors.surface_container_lowest);
    match (&frame.snapshot.phase, frame.preview) {
        (Phase::Live, Some(texture)) => live_video(ui, rect, texture, frame),
        (Phase::Live, None) => progress(ui, rect, "Starting camera…", &frame.form.camera.device),
        (Phase::Loading, _) => progress(ui, rect, "Loading models…", "Warming up the inference engine"),
        (Phase::Failed(error), _) => failed(ui, rect, error),
        (Phase::Ready, _) => idle(ui, rect, frame),
    }
    if matches!(frame.snapshot.phase, Phase::Ready | Phase::Live) {
        start_stop_fab(ui, rect, frame);
    }
}

/// Place content of unknown height centred in `rect`, using last frame's measured height.
fn centered(ui: &mut Ui, rect: Rect, id: &str, add: impl FnOnce(&mut Ui)) {
    let id = Id::new(id);
    let height: f32 = ui.ctx().data(|d| d.get_temp(id)).unwrap_or_default();
    let top = rect.center().y - height / 2.0;
    let child = Rect::from_min_max(pos2(rect.left(), top.max(rect.top())), rect.max);
    let used = ui.scope_builder(UiBuilder::new().max_rect(child).layout(Layout::top_down(Align::Center)), add).response.rect;
    if (used.height() - height).abs() > 0.5 {
        ui.ctx().data_mut(|d| d.insert_temp(id, used.height()));
        ui.ctx().request_repaint();
    }
}

fn progress(ui: &mut Ui, rect: Rect, title: &str, body: &str) {
    let colors = theme::colors(ui.ctx());
    centered(ui, rect, "preview-progress", |ui| {
        ui.add(CircularProgress::new(PROGRESS_SIZE));
        ui.add_space(spacing::XL);
        ui.label(TypeRole::TitleLarge.rich(title, colors.on_surface));
        ui.label(TypeRole::BodyMedium.rich(body, colors.on_surface_variant));
    });
}

fn failed(ui: &mut Ui, rect: Rect, error: &str) {
    centered(ui, rect, "preview-failed", |ui| {
        empty_state(ui, icons::ERROR, "Something went wrong", error, |_| {});
    });
}

fn idle(ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>) {
    let has_source = frame.snapshot.has_source;
    let intake = frame.intake;
    let mut pick = false;
    centered(ui, rect, "preview-idle", |ui| {
        if has_source {
            empty_state(ui, icons::VIDEOCAM, "Ready to go live", "Press Start or hit Space to open the camera and begin swapping.", |_| {});
        } else {
            let body = if intake.drop {
                "Drop a clear, front-facing photo anywhere in the window to choose whose face to wear."
            } else {
                "Choose a clear, front-facing photo of the face you want to wear."
            };
            empty_state(ui, icons::FRAME_PERSON, "Add a source face", body, |ui| {
                if intake.pick {
                    pick = ui.add(Button::filled("Choose photo").icon(icons::FOLDER_OPEN)).clicked();
                }
            });
        }
    });
    if pick {
        frame.actions.push(Action::PickSource);
    }
}

fn start_stop_fab(ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>) {
    let live = frame.is_live();
    let (icon, label, color) = if live { (icons::STOP, "Stop", FabColor::Error) } else { (icons::PLAY_ARROW, "Start", FabColor::Primary) };
    let anchor = rect.right_bottom() - Vec2::splat(OVERLAY_INSET);
    let area = Rect::from_min_max(pos2(rect.left(), anchor.y - theme::size::TARGET * 2.0), anchor);
    let clicked = ui
        .scope_builder(UiBuilder::new().max_rect(area).layout(Layout::right_to_left(Align::Max)), |ui| {
            ui.add(ExtendedFab::new("start-stop", icon, label).color(color)).on_hover_text("Space").clicked()
        })
        .inner;
    if clicked {
        frame.actions.push(Action::ToggleLive);
    }
}

fn live_video(ui: &mut Ui, rect: Rect, texture: &TextureHandle, frame: &Frame<'_>) {
    let image = texture.size_vec2();
    let scale = (rect.size() / image).min_elem();
    let fitted = Rect::from_center_size(rect.center(), image * scale);
    eframe::egui::Image::new(texture).corner_radius(shape::EXTRA_LARGE).paint_at(ui, fitted);
    let inner = rect.shrink(OVERLAY_INSET);
    live_badge(ui, inner.left_top());
    stat_chips(ui, inner, frame);
}

/// The pulsing red "LIVE" pill in the top-left corner.
fn live_badge(ui: &mut Ui, top_left: eframe::egui::Pos2) {
    let colors = theme::colors(ui.ctx());
    let label_width = paint::text_width(ui, TypeRole::LabelLarge, "LIVE");
    let width = spacing::SM + LIVE_DOT + spacing::SM + label_width + spacing::MD;
    let rect = Rect::from_min_size(top_left, vec2(width, LIVE_BADGE_HEIGHT));
    let painter = ui.painter();
    painter.rect_filled(rect, shape::full(LIVE_BADGE_HEIGHT), colors.error);
    let time = ui.input(|i| i.time);
    let pulse = 0.5 + 0.5 * (time * std::f64::consts::TAU / LIVE_PULSE_PERIOD).cos() as f32;
    let dot_opacity = LIVE_PULSE_MIN_OPACITY + (1.0 - LIVE_PULSE_MIN_OPACITY) * pulse;
    let dot = pos2(rect.left() + spacing::SM + LIVE_DOT / 2.0, rect.center().y);
    painter.circle_filled(dot, LIVE_DOT / 2.0, colors.on_error.gamma_multiply(dot_opacity));
    let text_pos = pos2(dot.x + LIVE_DOT / 2.0 + spacing::SM, rect.center().y);
    paint::text(ui, text_pos, Align2::LEFT_CENTER, TypeRole::LabelLarge, "LIVE", colors.on_error);
    ui.ctx().request_repaint();
}

/// FPS / latency / faces in the top-right, per-stage timings in the bottom-left.
fn stat_chips(ui: &mut Ui, inner: Rect, frame: &Frame<'_>) {
    let stats = &frame.snapshot.stats;
    let top = Rect::from_min_max(inner.center_top(), inner.right_top() + vec2(0.0, theme::size::TARGET));
    ui.scope_builder(UiBuilder::new().max_rect(top).layout(Layout::right_to_left(Align::Min)), |ui| {
        ui.spacing_mut().item_spacing.x = spacing::SM;
        let faces = if stats.faces == 1 { "1 face".to_owned() } else { format!("{} faces", stats.faces) };
        ui.add(Chip::status(&faces, false).icon(icons::FACE));
        ui.add(Chip::status(&format!("{:.1} ms", stats.total_ms), false).icon(icons::TIMER));
        ui.add(Chip::status(&format!("{:.0} fps", frame.snapshot.fps), true).icon(icons::SPEED));
    });
    let bottom = Rect::from_min_max(inner.left_bottom() - vec2(0.0, theme::size::TARGET), inner.center_bottom());
    ui.scope_builder(UiBuilder::new().max_rect(bottom).layout(Layout::left_to_right(Align::Max)), |ui| {
        ui.spacing_mut().item_spacing.x = spacing::SM;
        let stages = [("Detect", stats.detect_ms), ("Swap", stats.swap_ms), ("Enhance", stats.enhance_ms)];
        for (stage, ms) in stages.into_iter().filter(|(_, ms)| *ms > 0.0) {
            ui.add(Chip::status(&format!("{stage} {ms:.1} ms"), false));
        }
    });
}
