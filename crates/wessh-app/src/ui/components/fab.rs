//! MD3 extended floating action button.

use super::paint;
use crate::ui::theme::{self, ColorScheme, Elevation, TypeRole, motion, shape, size, spacing};
use eframe::egui::{Align2, Color32, CursorIcon, Id, Response, Sense, Ui, Vec2, Widget, WidgetInfo, WidgetType, pos2};

const HEIGHT: f32 = 56.0;
const LEADING_PADDING: f32 = spacing::LG;
const TRAILING_PADDING: f32 = 20.0;
const ICON_GAP: f32 = spacing::MD;

/// Container colour family of a FAB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FabColor {
    Primary,
    Secondary,
    Tertiary,
    Error,
}

impl FabColor {
    fn roles(self, c: &ColorScheme) -> (Color32, Color32) {
        match self {
            Self::Primary => (c.primary_container, c.on_primary_container),
            Self::Secondary => (c.secondary_container, c.on_secondary_container),
            Self::Tertiary => (c.tertiary_container, c.on_tertiary_container),
            Self::Error => (c.error_container, c.on_error_container),
        }
    }
}

/// Icon + label FAB. Colour and width animate when they change between frames.
#[must_use = "add it with `ui.add(...)`"]
pub struct ExtendedFab<'a> {
    id: Id,
    icon: char,
    label: &'a str,
    color: FabColor,
}

impl<'a> ExtendedFab<'a> {
    /// A FAB; `id_salt` keeps its animations stable while its label changes.
    pub fn new(id_salt: impl std::hash::Hash + std::fmt::Debug, icon: char, label: &'a str) -> Self {
        Self { id: Id::new(id_salt), icon, label, color: FabColor::Primary }
    }

    /// Container colour family.
    pub fn color(mut self, color: FabColor) -> Self {
        self.color = color;
        self
    }
}

impl Widget for ExtendedFab<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let enabled = ui.is_enabled();
        let colors = theme::colors(ui.ctx());
        let label_width = paint::text_width(ui, TypeRole::LabelLarge, self.label);
        let target_width = LEADING_PADDING + size::ICON + ICON_GAP + label_width + TRAILING_PADDING;
        let width = ui.ctx().animate_value_with_time(self.id.with("width"), target_width, motion::SHORT_4);
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, HEIGHT), sense);
        let response = response.on_hover_cursor(if enabled { CursorIcon::PointingHand } else { CursorIcon::Default });
        response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, self.label));

        let (container, content) = self.color.roles(&colors);
        let container = paint::animate_color(ui, self.id.with("container"), container, motion::MEDIUM_2);
        let content = paint::animate_color(ui, self.id.with("content"), content, motion::MEDIUM_2);
        let (container, content) =
            if enabled { (container, content) } else { (paint::container(colors.on_surface, false), colors.on_surface) };
        let elevation = if response.hovered() && enabled { Elevation::Level4 } else { Elevation::Level3 };
        let radius = shape::LARGE;
        let painter = ui.painter();
        if enabled {
            painter.add(elevation.shadow(colors.shadow).as_shape(rect, radius));
        }
        painter.rect_filled(rect, radius, container);
        paint::state_layer(ui, &response, rect, radius, content);
        paint::focus_ring(ui, &response, rect, radius, &colors);

        let icon_center = pos2(rect.left() + LEADING_PADDING + size::ICON / 2.0, rect.center().y);
        paint::icon(painter, icon_center, self.icon, size::ICON, content, true);
        let label_pos = pos2(icon_center.x + size::ICON / 2.0 + ICON_GAP, rect.center().y);
        let clip = ui.painter().with_clip_rect(rect);
        let galley = TypeRole::LabelLarge.galley(ui, self.label, content);
        clip.galley(Align2::LEFT_CENTER.anchor_size(label_pos, galley.size()).min, galley, content);
        response
    }
}
