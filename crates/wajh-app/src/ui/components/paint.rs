//! Painting helpers shared by all components: icons, text, state layers, focus rings.

use crate::ui::theme::{self, ColorScheme, TypeRole, motion, state};
use eframe::egui::{Align2, Color32, CornerRadius, Painter, Pos2, Rect, Response, Stroke, StrokeKind, Ui, Vec2};

/// Thickness of the MD3 focus indicator.
const FOCUS_RING_WIDTH: f32 = 3.0;
/// Gap between a component and its focus indicator.
const FOCUS_RING_OFFSET: f32 = 2.0;

/// Paint a Material Symbol centred on `center`.
pub fn icon(painter: &Painter, center: Pos2, glyph: char, size: f32, color: Color32, filled: bool) {
    painter.text(center, Align2::CENTER_CENTER, glyph, theme::icon_font(size, filled), color);
}

/// Paint `text` in a type role, anchored at `pos` with `align`. Returns the text's rect.
pub fn text(ui: &Ui, pos: Pos2, align: Align2, role: TypeRole, text: &str, color: Color32) -> Rect {
    let galley = role.galley(ui, text, color);
    let rect = align.anchor_size(pos, galley.size());
    ui.painter().galley(rect.min, galley, color);
    rect
}

/// Width `text` occupies in a type role.
#[must_use]
pub fn text_width(ui: &Ui, role: TypeRole, text: &str) -> f32 {
    role.galley(ui, text, Color32::PLACEHOLDER).size().x
}

/// Animated MD3 state-layer opacity for an interactive response
/// (hover 8 %, focus 10 %, pressed 10 %, dragged 16 %).
#[must_use]
pub fn state_opacity(ui: &Ui, response: &Response) -> f32 {
    let target = if !response.enabled() {
        0.0
    } else if response.dragged() {
        state::DRAGGED
    } else if response.is_pointer_button_down_on() {
        state::PRESSED
    } else if response.has_focus() {
        state::FOCUS
    } else if response.hovered() {
        state::HOVER
    } else {
        0.0
    };
    ui.ctx().animate_value_with_time(response.id.with("state-layer"), target, motion::SHORT_2)
}

/// Paint a state layer of `content` colour over `rect`.
pub fn state_layer(ui: &Ui, response: &Response, rect: Rect, radius: CornerRadius, content: Color32) {
    let opacity = state_opacity(ui, response);
    if opacity > 0.0 {
        ui.painter().rect_filled(rect, radius, content.gamma_multiply(opacity));
    }
}

/// Paint the keyboard focus indicator around `rect` when focused.
pub fn focus_ring(ui: &Ui, response: &Response, rect: Rect, radius: CornerRadius, colors: &ColorScheme) {
    if response.has_focus() && ui.memory(|m| m.focused().is_some()) && ui.input(|i| !i.pointer.any_down()) {
        let expanded = rect.expand(FOCUS_RING_OFFSET + FOCUS_RING_WIDTH / 2.0);
        let radius = radius + (FOCUS_RING_OFFSET + FOCUS_RING_WIDTH / 2.0) as u8;
        ui.painter().rect_stroke(expanded, radius, Stroke::new(FOCUS_RING_WIDTH, colors.secondary), StrokeKind::Middle);
    }
}

/// A container colour for the given enabled state.
///
/// egui already fades everything in a disabled `Ui` by `disabled_alpha`, which the theme sets
/// to the MD3 disabled-content opacity, so disabled content needs no extra fading. Containers
/// should end at the lower MD3 disabled-container opacity, so they get the remaining factor.
#[must_use]
pub fn container(color: Color32, enabled: bool) -> Color32 {
    if enabled { color } else { state::with_opacity(color, state::DISABLED_CONTAINER / state::DISABLED_CONTENT) }
}

/// Smoothly transition towards `target` over `duration` seconds (per channel).
#[must_use]
pub fn animate_color(ui: &Ui, id: eframe::egui::Id, target: Color32, duration: f32) -> Color32 {
    let ctx = ui.ctx();
    let [r, g, b, a] = target.to_array().map(f32::from);
    let channel = |salt: &str, value: f32| ctx.animate_value_with_time(id.with(salt), value, duration).round() as u8;
    Color32::from_rgba_premultiplied(channel("r", r), channel("g", g), channel("b", b), channel("a", a))
}

/// A rect of `size` centred in `outer`.
#[must_use]
pub fn centered(outer: Rect, size: Vec2) -> Rect {
    Rect::from_center_size(outer.center(), size)
}
