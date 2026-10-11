//! MD3 chips: assist chips, filter chips, and read-only status chips.

use super::paint;
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{Align2, Color32, CursorIcon, Response, Sense, Stroke, StrokeKind, Ui, Vec2, Widget, WidgetInfo, WidgetType, pos2};

/// Height of every chip.
pub const HEIGHT: f32 = 32.0;
const PADDING: f32 = spacing::LG;
const PADDING_WITH_ICON: f32 = spacing::SM;
const ICON_GAP: f32 = spacing::SM;

/// What kind of chip to draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChipKind {
    /// Outlined action chip with a primary-coloured icon.
    Assist,
    /// Toggleable chip; shows a check mark and a secondary container when selected.
    Filter { selected: bool },
    /// Non-interactive tonal chip for live status readouts.
    Status { emphasis: bool },
}

/// A compact chip with optional leading icon.
#[must_use = "add it with `ui.add(...)`"]
pub struct Chip<'a> {
    kind: ChipKind,
    label: &'a str,
    icon: Option<char>,
}

impl<'a> Chip<'a> {
    /// An assist chip: a smart action shown as an outlined chip.
    pub fn assist(label: &'a str) -> Self {
        Self { kind: ChipKind::Assist, label, icon: None }
    }

    /// A filter chip in the given selection state.
    pub fn filter(label: &'a str, selected: bool) -> Self {
        Self { kind: ChipKind::Filter { selected }, label, icon: None }
    }

    /// A read-only status chip; `emphasis` uses the tertiary container.
    pub fn status(label: &'a str, emphasis: bool) -> Self {
        Self { kind: ChipKind::Status { emphasis }, label, icon: None }
    }

    /// Add a leading icon.
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }
}

impl Widget for Chip<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let colors = theme::colors(ui.ctx());
        let selected = matches!(self.kind, ChipKind::Filter { selected: true });
        let interactive = !matches!(self.kind, ChipKind::Status { .. });
        let icon = if selected { Some(icons::CHECK) } else { self.icon };
        let leading = if icon.is_some() { PADDING_WITH_ICON + size::ICON_SMALL + ICON_GAP } else { PADDING };
        let width = leading + paint::text_width(ui, TypeRole::LabelLarge, self.label) + PADDING;
        let sense = if interactive { Sense::click() } else { Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, HEIGHT), sense);
        let response = if interactive { response.on_hover_cursor(CursorIcon::PointingHand) } else { response };
        response.widget_info(|| WidgetInfo::selected(WidgetType::Button, true, selected, self.label));

        let radius = shape::SMALL;
        let (fill, outline, label_color, icon_color) = match self.kind {
            ChipKind::Assist => (Color32::TRANSPARENT, Some(colors.outline_variant), colors.on_surface, colors.primary),
            ChipKind::Filter { selected } => {
                let t = ui.ctx().animate_bool_with_time(response.id, selected, motion::SHORT_4);
                let fill = colors.secondary_container.gamma_multiply(t);
                let label = colors.on_surface_variant.lerp_to_gamma(colors.on_secondary_container, t);
                let outline = (t < 1.0).then(|| colors.outline_variant.gamma_multiply(1.0 - t));
                (fill, outline, label, label)
            }
            ChipKind::Status { emphasis: true } => {
                (colors.tertiary_container, None, colors.on_tertiary_container, colors.on_tertiary_container)
            }
            ChipKind::Status { emphasis: false } => (colors.surface_container_high, None, colors.on_surface, colors.on_surface_variant),
        };
        let painter = ui.painter();
        painter.rect_filled(rect, radius, fill);
        if let Some(outline) = outline {
            painter.rect_stroke(rect, radius, Stroke::new(size::STROKE, outline), StrokeKind::Inside);
        }
        if interactive {
            paint::state_layer(ui, &response, rect, radius, label_color);
            paint::focus_ring(ui, &response, rect, radius, &colors);
        }
        if let Some(glyph) = icon {
            let center = pos2(rect.left() + PADDING_WITH_ICON + size::ICON_SMALL / 2.0, rect.center().y);
            paint::icon(painter, center, glyph, size::ICON_SMALL, icon_color, selected);
        }
        paint::text(ui, pos2(rect.left() + leading, rect.center().y), Align2::LEFT_CENTER, TypeRole::LabelLarge, self.label, label_color);
        response
    }
}
