//! MD3 common buttons (filled, tonal, elevated, outlined, text) and icon buttons.

use super::paint;
use crate::ui::theme::{self, ColorScheme, Elevation, TypeRole, shape, size, spacing};
use eframe::egui::{Align2, Color32, CursorIcon, Response, Sense, Stroke, StrokeKind, Ui, Vec2, Widget, WidgetInfo, WidgetType};

const HEIGHT: f32 = 40.0;
const PADDING: f32 = spacing::XL;
const PADDING_WITH_ICON: f32 = spacing::LG;
const TEXT_PADDING: f32 = spacing::MD;
const ICON_GAP: f32 = spacing::SM;
const ICON_BUTTON_SIZE: f32 = 40.0;

/// Emphasis level of a common button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonVariant {
    Filled,
    Tonal,
    Elevated,
    Outlined,
    Text,
}

struct ButtonColors {
    container: Color32,
    content: Color32,
    outline: Option<Color32>,
    elevation: Elevation,
}

impl ButtonVariant {
    fn colors(self, c: &ColorScheme) -> ButtonColors {
        let (container, content, outline, elevation) = match self {
            Self::Filled => (c.primary, c.on_primary, None, Elevation::Level0),
            Self::Tonal => (c.secondary_container, c.on_secondary_container, None, Elevation::Level0),
            Self::Elevated => (c.surface_container_low, c.primary, None, Elevation::Level1),
            Self::Outlined => (Color32::TRANSPARENT, c.primary, Some(c.outline), Elevation::Level0),
            Self::Text => (Color32::TRANSPARENT, c.primary, None, Elevation::Level0),
        };
        ButtonColors { container, content, outline, elevation }
    }

    fn has_container(self) -> bool {
        matches!(self, Self::Filled | Self::Tonal | Self::Elevated)
    }
}

/// A labelled button with an optional leading icon.
#[must_use = "add it with `ui.add(...)`"]
pub struct Button<'a> {
    variant: ButtonVariant,
    label: &'a str,
    icon: Option<char>,
}

impl<'a> Button<'a> {
    /// A button of the given emphasis.
    pub fn new(variant: ButtonVariant, label: &'a str) -> Self {
        Self { variant, label, icon: None }
    }

    /// Highest emphasis: primary actions.
    pub fn filled(label: &'a str) -> Self {
        Self::new(ButtonVariant::Filled, label)
    }

    /// Medium emphasis on a secondary container.
    pub fn tonal(label: &'a str) -> Self {
        Self::new(ButtonVariant::Tonal, label)
    }

    /// Medium emphasis with an outline.
    pub fn outlined(label: &'a str) -> Self {
        Self::new(ButtonVariant::Outlined, label)
    }

    /// Lowest emphasis.
    pub fn text(label: &'a str) -> Self {
        Self::new(ButtonVariant::Text, label)
    }

    /// Add a leading icon.
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }

    fn padding(&self) -> (f32, f32) {
        match (self.variant, self.icon) {
            (ButtonVariant::Text, None) => (TEXT_PADDING, TEXT_PADDING),
            (ButtonVariant::Text, Some(_)) => (TEXT_PADDING, PADDING_WITH_ICON),
            (_, Some(_)) => (PADDING_WITH_ICON, PADDING),
            (_, None) => (PADDING, PADDING),
        }
    }
}

impl Widget for Button<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let enabled = ui.is_enabled();
        let colors = theme::colors(ui.ctx());
        let (lead, trail) = self.padding();
        let icon_width = if self.icon.is_some() { size::ICON_SMALL + ICON_GAP } else { 0.0 };
        let label_width = paint::text_width(ui, TypeRole::LabelLarge, self.label);
        let desired = Vec2::new(lead + icon_width + label_width + trail, HEIGHT);
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(desired, sense);
        let response = response.on_hover_cursor(if enabled { CursorIcon::PointingHand } else { CursorIcon::Default });
        response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, self.label));
        if !ui.is_rect_visible(rect) {
            return response;
        }

        let palette = self.variant.colors(&colors);
        let radius = shape::full(HEIGHT);
        let painter = ui.painter();
        if self.variant.has_container() {
            let fill = if enabled { palette.container } else { paint::container(colors.on_surface, false) };
            if enabled {
                painter.add(palette.elevation.shadow(colors.shadow).as_shape(rect, radius));
            }
            painter.rect_filled(rect, radius, fill);
        }
        if let Some(outline) = palette.outline {
            let stroke_color = if enabled { outline } else { paint::container(colors.on_surface, false) };
            painter.rect_stroke(rect, radius, Stroke::new(size::STROKE, stroke_color), StrokeKind::Inside);
        }
        let content = if enabled { palette.content } else { paint::content(colors.on_surface, false) };
        paint::state_layer(ui, &response, rect, radius, content);
        paint::focus_ring(ui, &response, rect, radius, &colors);

        let mut x = rect.left() + lead;
        if let Some(icon) = self.icon {
            paint::icon(painter, eframe::egui::pos2(x + size::ICON_SMALL / 2.0, rect.center().y), icon, size::ICON_SMALL, content, false);
            x += icon_width;
        }
        paint::text(ui, eframe::egui::pos2(x, rect.center().y), Align2::LEFT_CENTER, TypeRole::LabelLarge, self.label, content);
        response
    }
}

/// A 40 dp circular icon button (standard or filled-tonal).
#[must_use = "add it with `ui.add(...)`"]
pub struct IconButton<'a> {
    icon: char,
    tooltip: &'a str,
    style: IconButtonStyle,
    selected: bool,
}

/// Container style of an icon button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IconButtonStyle {
    /// No container; `onSurfaceVariant` icon.
    Standard,
    /// `secondaryContainer` circle.
    Tonal,
    /// No container, for inverse surfaces such as snackbars.
    Inverse,
}

impl<'a> IconButton<'a> {
    /// A standard (container-less) icon button with a tooltip.
    pub fn new(icon: char, tooltip: &'a str) -> Self {
        Self { icon, tooltip, style: IconButtonStyle::Standard, selected: false }
    }

    /// Use the filled-tonal style.
    pub fn tonal(mut self) -> Self {
        self.style = IconButtonStyle::Tonal;
        self
    }

    /// Style for inverse surfaces such as snackbars.
    pub fn inverse(mut self) -> Self {
        self.style = IconButtonStyle::Inverse;
        self
    }

    /// Show the toggle-selected state (filled icon in primary).
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

impl Widget for IconButton<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let colors = theme::colors(ui.ctx());
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(ICON_BUTTON_SIZE), Sense::click());
        let response = response.on_hover_cursor(CursorIcon::PointingHand).on_hover_text(self.tooltip);
        response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, self.tooltip));
        let radius = shape::full(ICON_BUTTON_SIZE);
        let content = match (self.style, self.selected) {
            (IconButtonStyle::Tonal, _) => colors.on_secondary_container,
            (IconButtonStyle::Inverse, _) => colors.inverse_on_surface,
            (IconButtonStyle::Standard, true) => colors.primary,
            (IconButtonStyle::Standard, false) => colors.on_surface_variant,
        };
        if self.style == IconButtonStyle::Tonal {
            ui.painter().rect_filled(rect, radius, colors.secondary_container);
        }
        paint::state_layer(ui, &response, rect, radius, content);
        paint::focus_ring(ui, &response, rect, radius, &colors);
        paint::icon(ui.painter(), rect.center(), self.icon, size::ICON, content, self.selected);
        response
    }
}
