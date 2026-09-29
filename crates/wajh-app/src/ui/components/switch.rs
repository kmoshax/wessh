//! MD3 switch.

use super::paint;
use crate::ui::icons;
use crate::ui::theme::{self, motion, shape, size};
use eframe::egui::{CursorIcon, Response, Sense, Stroke, StrokeKind, Ui, Vec2, Widget, WidgetInfo, WidgetType, lerp, pos2};

const TRACK: Vec2 = Vec2::new(52.0, 32.0);
const TRACK_OUTLINE: f32 = 2.0;
const HANDLE_OFF: f32 = 16.0;
const HANDLE_ON: f32 = 24.0;
const HANDLE_PRESSED: f32 = 28.0;
const STATE_LAYER: f32 = 40.0;
const ICON_SIZE: f32 = 16.0;

/// An on/off switch bound to a `bool`.
#[must_use = "add it with `ui.add(...)`"]
pub struct Switch<'a> {
    on: &'a mut bool,
}

impl<'a> Switch<'a> {
    /// A switch toggling `on`.
    pub fn new(on: &'a mut bool) -> Self {
        Self { on }
    }
}

impl Widget for Switch<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let enabled = ui.is_enabled();
        let colors = theme::colors(ui.ctx());
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, mut response) = ui.allocate_exact_size(Vec2::new(TRACK.x, size::TARGET), sense);
        if response.clicked() {
            *self.on = !*self.on;
            response.mark_changed();
        }
        let on = *self.on;
        response.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, enabled, on, ""));
        let response = response.on_hover_cursor(if enabled { CursorIcon::PointingHand } else { CursorIcon::Default });

        let t = ui.ctx().animate_bool_with_time_and_easing(response.id, on, motion::SHORT_4, motion::emphasized);
        let track = paint::centered(rect, TRACK);
        let radius = shape::full(TRACK.y);
        let hovered = response.hovered() && enabled;
        let pressed = response.is_pointer_button_down_on();

        let track_off = colors.surface_container_highest;
        let track_fill = match (enabled, on) {
            (true, _) => track_off.lerp_to_gamma(colors.primary, t),
            (false, true) => paint::container(colors.on_surface, false),
            (false, false) => paint::container(track_off, false),
        };
        let painter = ui.painter();
        painter.rect_filled(track, radius, track_fill);
        if t < 1.0 {
            let outline = paint::content(colors.outline, enabled).gamma_multiply(1.0 - t);
            painter.rect_stroke(track, radius, Stroke::new(TRACK_OUTLINE, outline), StrokeKind::Inside);
        }

        let handle_size = if pressed { HANDLE_PRESSED } else { lerp(HANDLE_OFF..=HANDLE_ON, t) };
        let inset = TRACK.y / 2.0;
        let center = pos2(lerp(track.left() + inset..=track.right() - inset, t), track.center().y);
        let handle_off = if hovered { colors.on_surface_variant } else { colors.outline };
        let handle_on = if hovered { colors.primary_container } else { colors.on_primary };
        let handle = if enabled {
            handle_off.lerp_to_gamma(handle_on, t)
        } else if on {
            colors.surface
        } else {
            paint::content(colors.on_surface, false)
        };
        let layer_rect = eframe::egui::Rect::from_center_size(center, Vec2::splat(STATE_LAYER));
        let layer_color = if on { colors.primary } else { colors.on_surface };
        paint::state_layer(ui, &response, layer_rect, shape::full(STATE_LAYER), layer_color);
        paint::focus_ring(ui, &response, track, radius, &colors);
        painter.circle_filled(center, handle_size / 2.0, handle);
        if t > 0.5 {
            let icon_color = paint::content(colors.on_primary_container, enabled).gamma_multiply((t - 0.5) * 2.0);
            paint::icon(painter, center, icons::CHECK, ICON_SIZE, icon_color, false);
        }
        response
    }
}
