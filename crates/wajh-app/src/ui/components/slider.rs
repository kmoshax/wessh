//! MD3 slider (2024 spec: thick track, bar handle, value indicator while dragging).

use super::paint;
use crate::ui::theme::{self, TypeRole, motion, shape};
use eframe::egui::{Align2, CornerRadius, CursorIcon, Key, Rect, Response, Sense, Ui, Vec2, Widget, WidgetInfo, emath, pos2, vec2};
use std::ops::RangeInclusive;

const HEIGHT: f32 = 44.0;
const TRACK_HEIGHT: f32 = 16.0;
const HANDLE_WIDTH: f32 = 4.0;
const HANDLE_WIDTH_PRESSED: f32 = 2.0;
const HANDLE_GAP: f32 = 6.0;
const INNER_CORNER: u8 = 2;
const STOP_DOT: f32 = 4.0;
const INDICATOR_SIZE: Vec2 = Vec2::new(48.0, 44.0);
const INDICATOR_GAP: f32 = 4.0;
/// Fraction of the range moved by one arrow-key press.
const KEY_STEP_FRACTION: f32 = 0.05;

/// A continuous slider over `range`.
#[must_use = "add it with `ui.add(...)`"]
pub struct Slider<'a> {
    value: &'a mut f32,
    range: RangeInclusive<f32>,
    format: fn(f32) -> String,
}

impl<'a> Slider<'a> {
    /// A slider editing `value` within `range`.
    pub fn new(value: &'a mut f32, range: RangeInclusive<f32>) -> Self {
        Self { value, range, format: |v| format!("{v:.2}") }
    }

    /// How the value indicator formats the value.
    pub fn format(mut self, format: fn(f32) -> String) -> Self {
        self.format = format;
        self
    }
}

impl Widget for Slider<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let enabled = ui.is_enabled();
        let colors = theme::colors(ui.ctx());
        let width = ui.available_width();
        let sense = if enabled { Sense::click_and_drag() } else { Sense::hover() };
        let (rect, mut response) = ui.allocate_exact_size(vec2(width, HEIGHT), sense);
        let response_hover = if enabled { CursorIcon::Grab } else { CursorIcon::Default };
        response = response.on_hover_cursor(response_hover);
        let track = Rect::from_center_size(rect.center(), vec2(width, TRACK_HEIGHT));
        let (min, max) = (*self.range.start(), *self.range.end());

        if let Some(pointer) = response.interact_pointer_pos() {
            let t = emath::remap_clamp(pointer.x, track.x_range(), 0.0..=1.0);
            set(self.value, emath::lerp(min..=max, t), &self.range, &mut response);
        }
        if response.has_focus() {
            let step = (max - min) * KEY_STEP_FRACTION;
            let delta =
                ui.input(|i| f32::from(u8::from(i.key_pressed(Key::ArrowRight))) - f32::from(u8::from(i.key_pressed(Key::ArrowLeft))));
            if delta != 0.0 {
                set(self.value, *self.value + delta * step, &self.range, &mut response);
            }
        }
        let value = *self.value;
        response.widget_info(|| WidgetInfo::slider(enabled, f64::from(value), ""));

        let t = emath::remap_clamp(value, min..=max, 0.0..=1.0);
        let handle_x = emath::lerp(track.x_range(), t);
        let active = paint::content(colors.primary, enabled);
        let inactive = if enabled { colors.secondary_container } else { paint::container(colors.on_surface, false) };
        let painter = ui.painter();
        let outer = shape::full(TRACK_HEIGHT);

        let active_right = handle_x - HANDLE_GAP;
        if active_right > track.left() {
            let segment = Rect::from_x_y_ranges(track.left()..=active_right, track.y_range());
            painter.rect_filled(segment, CornerRadius { ne: INNER_CORNER, se: INNER_CORNER, ..outer }, active);
        }
        let inactive_left = handle_x + HANDLE_GAP;
        if inactive_left < track.right() {
            let segment = Rect::from_x_y_ranges(inactive_left..=track.right(), track.y_range());
            painter.rect_filled(segment, CornerRadius { nw: INNER_CORNER, sw: INNER_CORNER, ..outer }, inactive);
            let dot = pos2(track.right() - TRACK_HEIGHT / 2.0, track.center().y);
            painter.circle_filled(dot, STOP_DOT / 2.0, active);
        }

        let pressed = response.dragged() || response.is_pointer_button_down_on();
        let handle_width = if pressed { HANDLE_WIDTH_PRESSED } else { HANDLE_WIDTH };
        let handle = Rect::from_center_size(pos2(handle_x, rect.center().y), vec2(handle_width, HEIGHT));
        painter.rect_filled(handle, shape::full(handle_width), active);
        paint::focus_ring(ui, &response, handle, shape::full(handle_width), &colors);

        let shown = ui.ctx().animate_bool_with_time(response.id.with("indicator"), pressed, motion::SHORT_2);
        if shown > 0.0 {
            value_indicator(ui, handle, &(self.format)(value), shown);
        }
        response
    }
}

fn set(value: &mut f32, new: f32, range: &RangeInclusive<f32>, response: &mut Response) {
    let clamped = new.clamp(*range.start(), *range.end());
    if clamped != *value {
        *value = clamped;
        response.mark_changed();
    }
}

/// The inverse-surface bubble above the handle showing the current value.
fn value_indicator(ui: &Ui, handle: Rect, label: &str, opacity: f32) {
    let colors = theme::colors(ui.ctx());
    let rect = Rect::from_center_size(pos2(handle.center().x, handle.top() - INDICATOR_GAP - INDICATOR_SIZE.y / 2.0), INDICATOR_SIZE);
    let painter = ui.ctx().layer_painter(eframe::egui::LayerId::new(eframe::egui::Order::Tooltip, ui.id().with("slider-value")));
    painter.rect_filled(rect, shape::full(INDICATOR_SIZE.y), colors.inverse_surface.gamma_multiply(opacity));
    let galley = TypeRole::LabelLarge.galley(ui, label, colors.inverse_on_surface.gamma_multiply(opacity));
    painter.galley(Align2::CENTER_CENTER.anchor_size(rect.center(), galley.size()).min, galley, colors.inverse_on_surface);
}
