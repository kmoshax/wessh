//! MD3 single-select segmented button.

use super::paint;
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{
    Align2, CornerRadius, CursorIcon, Rect, Response, Sense, Stroke, StrokeKind, Ui, Widget, WidgetInfo, WidgetType, pos2, vec2,
};

const HEIGHT: f32 = 40.0;

/// Equal-width segments filling the available width; exactly one is selected.
#[must_use = "add it with `ui.add(...)`"]
pub struct SegmentedButton<'a, T> {
    value: &'a mut T,
    options: &'a [(T, &'a str)],
}

impl<'a, T: PartialEq + Copy> SegmentedButton<'a, T> {
    /// Segments for `options` (value, label), editing `value`.
    pub fn new(value: &'a mut T, options: &'a [(T, &'a str)]) -> Self {
        Self { value, options }
    }
}

impl<T: PartialEq + Copy> Widget for SegmentedButton<'_, T> {
    fn ui(self, ui: &mut Ui) -> Response {
        let enabled = ui.is_enabled();
        let colors = theme::colors(ui.ctx());
        let (outer, mut response) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
        let count = self.options.len().max(1);
        let segment_width = outer.width() / count as f32;
        let full = shape::full(HEIGHT);
        let outline = Stroke::new(size::STROKE, paint::content(colors.outline, enabled));

        for (index, (option, label)) in self.options.iter().enumerate() {
            let left = outer.left() + segment_width * index as f32;
            let rect = Rect::from_x_y_ranges(left..=left + segment_width, outer.y_range());
            let radius = segment_radius(index, count, full);
            let sense = if enabled { Sense::click() } else { Sense::hover() };
            let segment = ui.interact(rect, response.id.with(index), sense).on_hover_cursor(CursorIcon::PointingHand);
            let selected = *self.value == *option;
            segment.widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, enabled, selected, *label));
            if segment.clicked() && !selected {
                *self.value = *option;
                response.mark_changed();
            }

            let t = ui.ctx().animate_bool_with_time(segment.id, selected, motion::SHORT_4);
            let painter = ui.painter();
            if t > 0.0 {
                let fill = paint::container(colors.secondary_container, enabled).gamma_multiply(t);
                painter.rect_filled(rect, radius, fill);
            }
            let content = paint::content(colors.on_surface.lerp_to_gamma(colors.on_secondary_container, t), enabled);
            paint::state_layer(ui, &segment, rect, radius, content);
            paint::focus_ring(ui, &segment, rect, radius, &colors);
            segment_label(ui, rect, label, content, t);
        }
        let painter = ui.painter();
        painter.rect_stroke(outer, full, outline, StrokeKind::Inside);
        for index in 1..count {
            let x = outer.left() + segment_width * index as f32;
            painter.vline(x, outer.y_range(), outline);
        }
        response
    }
}

/// Only the outermost corners of the group are rounded.
fn segment_radius(index: usize, count: usize, full: CornerRadius) -> CornerRadius {
    let first = index == 0;
    let last = index + 1 == count;
    CornerRadius {
        nw: if first { full.nw } else { 0 },
        sw: if first { full.sw } else { 0 },
        ne: if last { full.ne } else { 0 },
        se: if last { full.se } else { 0 },
    }
}

/// Label, preceded by a check mark that grows in as the segment becomes selected.
fn segment_label(ui: &Ui, rect: Rect, label: &str, color: eframe::egui::Color32, selected: f32) {
    let icon_width = (size::ICON_SMALL + spacing::SM) * selected;
    let text_width = paint::text_width(ui, TypeRole::LabelLarge, label);
    let start = rect.center().x - f32::midpoint(icon_width, text_width);
    if selected > 0.0 {
        let center = pos2(start + size::ICON_SMALL / 2.0, rect.center().y);
        paint::icon(ui.painter(), center, icons::CHECK, size::ICON_SMALL * selected, color, false);
    }
    paint::text(ui, pos2(start + icon_width, rect.center().y), Align2::LEFT_CENTER, TypeRole::LabelLarge, label, color);
}
