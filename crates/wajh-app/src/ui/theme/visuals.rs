//! Maps MD3 colour roles and tokens onto egui's global [`Style`], so stock
//! widgets (text edits, scroll bars, tooltips, menus) match the custom components.

use super::scheme::ColorScheme;
use super::tokens::{Elevation, shape, size, spacing, state};
use super::typography;
use eframe::egui::style::{Selection, TextCursorStyle, WidgetVisuals, Widgets};
use eframe::egui::{Margin, Stroke, Style, Vec2, Visuals};

/// Opacity of the text-selection highlight.
const SELECTION_OPACITY: f32 = 0.4;
/// Width of scroll bars at rest.
const SCROLL_BAR_WIDTH: f32 = 6.0;

/// Build the full egui style for a colour scheme.
#[must_use]
pub fn style(colors: &ColorScheme) -> Style {
    let mut style = Style { visuals: visuals(colors), text_styles: typography::text_styles(), ..Style::default() };
    let spacing = &mut style.spacing;
    spacing.item_spacing = Vec2::new(spacing::SM, spacing::SM);
    spacing.button_padding = Vec2::new(spacing::LG, spacing::SM);
    spacing.menu_margin = Margin::symmetric(0, spacing::SM as i8);
    spacing.window_margin = Margin::same(spacing::XL as i8);
    spacing.interact_size = Vec2::new(size::TARGET, size::TARGET - spacing::SM);
    spacing.scroll.bar_width = SCROLL_BAR_WIDTH;
    spacing.scroll.floating = true;
    style.animation_time = super::tokens::motion::SHORT_4;
    style
}

fn visuals(c: &ColorScheme) -> Visuals {
    let base = if c.is_dark() { Visuals::dark() } else { Visuals::light() };
    Visuals {
        dark_mode: c.is_dark(),
        override_text_color: None,
        widgets: widgets(c),
        selection: Selection { bg_fill: c.primary.gamma_multiply(SELECTION_OPACITY), stroke: Stroke::new(size::STROKE, c.on_surface) },
        hyperlink_color: c.primary,
        faint_bg_color: c.surface_container_low,
        extreme_bg_color: c.surface_container_highest,
        text_edit_bg_color: Some(c.surface_container_highest),
        code_bg_color: c.surface_container_high,
        warn_fg_color: c.tertiary,
        error_fg_color: c.error,
        window_corner_radius: shape::EXTRA_LARGE,
        window_shadow: Elevation::Level3.shadow(c.shadow),
        window_fill: c.surface_container_high,
        window_stroke: Stroke::NONE,
        menu_corner_radius: shape::EXTRA_SMALL,
        panel_fill: c.surface,
        popup_shadow: Elevation::Level2.shadow(c.shadow),
        text_cursor: TextCursorStyle { stroke: Stroke::new(size::FOCUS_STROKE, c.primary), ..base.text_cursor },
        button_frame: true,
        striped: false,
        slider_trailing_fill: true,
        disabled_alpha: state::DISABLED_CONTENT,
        ..base
    }
}

fn widgets(c: &ColorScheme) -> Widgets {
    let visual = |fill, stroke, text| WidgetVisuals {
        bg_fill: fill,
        weak_bg_fill: fill,
        bg_stroke: stroke,
        corner_radius: shape::SMALL,
        fg_stroke: Stroke::new(size::STROKE, text),
        expansion: 0.0,
    };
    let hover = state::overlay(c.surface_container_highest, c.on_surface, state::HOVER);
    let pressed = state::overlay(c.surface_container_highest, c.on_surface, state::PRESSED);
    Widgets {
        noninteractive: visual(c.surface, Stroke::new(size::STROKE, c.outline_variant), c.on_surface_variant),
        inactive: visual(c.surface_container_highest, Stroke::NONE, c.on_surface),
        hovered: visual(hover, Stroke::new(size::STROKE, c.outline), c.on_surface),
        active: visual(pressed, Stroke::new(size::FOCUS_STROKE, c.primary), c.on_surface),
        open: visual(c.surface_container_high, Stroke::new(size::STROKE, c.outline_variant), c.on_surface),
    }
}
