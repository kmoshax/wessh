//! Before/after comparison: two images of the same size overlaid, split by a draggable divider.

use super::{Chip, paint};
use crate::ui::icons;
use crate::ui::theme::{self, Elevation, motion, shape, size, spacing};
use eframe::egui::{
    Align, CornerRadius, CursorIcon, Image, Key, Layout, Rect, Response, Sense, Stroke, TextureHandle, Ui, UiBuilder, Vec2, emath, pos2,
    vec2,
};

/// Diameter of the round handle on the divider.
const KNOB: f32 = 40.0;
const DIVIDER_WIDTH: f32 = 2.0;
/// Fraction of the width moved by one arrow-key press.
const KEY_STEP: f32 = 0.05;
/// Inset of the Before / After labels from the image corners.
const LABEL_INSET: f32 = spacing::MD;
/// Rough width reserved for a label, to fade it out before the divider covers it.
const LABEL_ROOM: f32 = 96.0;

/// A split view: `before` left of the divider, `after` right of it.
#[must_use = "call `.show(ui, rect)`"]
pub struct Compare<'a> {
    before: &'a TextureHandle,
    after: &'a TextureHandle,
    split: &'a mut f32,
    radius: CornerRadius,
}

impl<'a> Compare<'a> {
    /// Compare `before` with `after`; `split` (`0..=1`) is where the divider sits.
    pub fn new(before: &'a TextureHandle, after: &'a TextureHandle, split: &'a mut f32) -> Self {
        Self { before, after, split, radius: shape::LARGE }
    }

    /// Fit both images into `rect` and handle dragging.
    pub fn show(self, ui: &mut Ui, rect: Rect) -> Response {
        let colors = theme::colors(ui.ctx());
        let image = self.after.size_vec2();
        let scale = (rect.size() / image).min_elem();
        let fitted = Rect::from_center_size(rect.center(), image * scale);
        let id = ui.id().with("compare");
        let response = ui.interact(fitted, id, Sense::click_and_drag()).on_hover_cursor(CursorIcon::ResizeHorizontal);
        if let Some(pointer) = response.interact_pointer_pos() {
            *self.split = emath::remap_clamp(pointer.x, fitted.x_range(), 0.0..=1.0);
        }
        if response.has_focus() {
            let delta =
                ui.input(|i| f32::from(u8::from(i.key_pressed(Key::ArrowRight))) - f32::from(u8::from(i.key_pressed(Key::ArrowLeft))));
            *self.split = (*self.split + delta * KEY_STEP).clamp(0.0, 1.0);
        }

        Image::new(self.after).corner_radius(self.radius).paint_at(ui, fitted);
        let x = emath::lerp(fitted.x_range(), *self.split);
        let left = Rect::from_min_max(fitted.min, pos2(x, fitted.bottom()));
        let mut clipped = ui.new_child(UiBuilder::new().max_rect(left));
        clipped.set_clip_rect(left.intersect(ui.clip_rect()));
        Image::new(self.before).corner_radius(self.radius).paint_at(&clipped, fitted);

        labels(ui, fitted, x);
        let painter = ui.painter();
        painter.vline(x, fitted.y_range(), Stroke::new(DIVIDER_WIDTH, colors.primary));
        let knob = Rect::from_center_size(pos2(x, fitted.center().y), Vec2::splat(KNOB));
        let dragging = ui.ctx().animate_bool_with_time(id.with("drag"), response.dragged(), motion::SHORT_4);
        let elevation = if dragging > 0.0 { Elevation::Level3 } else { Elevation::Level2 };
        painter.add(elevation.shadow(colors.shadow).as_shape(knob, shape::full(KNOB)));
        painter.circle_filled(knob.center(), KNOB / 2.0, colors.primary);
        paint::state_layer(ui, &response, knob, shape::full(KNOB), colors.on_primary);
        paint::focus_ring(ui, &response, knob, shape::full(KNOB), &colors);
        paint::icon(painter, knob.center(), icons::COMPARE_ARROWS, size::ICON, colors.on_primary, false);
        response
    }
}

/// "Before" / "After" chips in the top corners, faded out as the divider approaches them.
fn labels(ui: &mut Ui, fitted: Rect, divider: f32) {
    let top = fitted.top() + LABEL_INSET;
    let room_left = ((divider - fitted.left()) / LABEL_ROOM - 1.0).clamp(0.0, 1.0);
    let room_right = ((fitted.right() - divider) / LABEL_ROOM - 1.0).clamp(0.0, 1.0);
    let height = size::TARGET;
    let sides = [
        ("Before", room_left, Rect::from_min_size(pos2(fitted.left() + LABEL_INSET, top), vec2(fitted.width() / 2.0, height)), Align::Min),
        (
            "After",
            room_right,
            Rect::from_min_max(pos2(fitted.center().x, top), pos2(fitted.right() - LABEL_INSET, top + height)),
            Align::Max,
        ),
    ];
    for (label, opacity, rect, align) in sides {
        if opacity <= 0.0 {
            continue;
        }
        let layout = if align == Align::Min { Layout::left_to_right(Align::Min) } else { Layout::right_to_left(Align::Min) };
        ui.scope_builder(UiBuilder::new().max_rect(rect).layout(layout), |ui| {
            ui.set_opacity(opacity);
            ui.add(Chip::status(label, false));
        });
    }
}
