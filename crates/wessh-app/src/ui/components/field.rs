//! MD3 filled text field and the filled exposed dropdown menu built on the same shell.

use super::paint;
use crate::ui::icons;
use crate::ui::theme::{self, ColorScheme, Elevation, TypeRole, motion, shape, size, spacing};
use eframe::egui::{
    self, Align2, CornerRadius, CursorIcon, Frame, Margin, Popup, PopupCloseBehavior, Rect, Response, Sense, Ui, Widget, WidgetInfo,
    WidgetType, pos2, vec2,
};

const HEIGHT: f32 = 56.0;
const PADDING_X: f32 = spacing::LG;
const LABEL_TOP: f32 = spacing::SM;
const VALUE_TOP: f32 = 24.0;
const ICON_GAP: f32 = spacing::MD;
const INDICATOR: f32 = 1.0;
const INDICATOR_FOCUSED: f32 = 2.0;
const MENU_ITEM_HEIGHT: f32 = 48.0;
const MENU_PADDING_Y: i8 = spacing::SM as i8;
/// Filled fields only round their top corners.
const TOP_ROUNDED: CornerRadius = CornerRadius { nw: 4, ne: 4, sw: 0, se: 0 };

/// Paint the field container, label, active indicator and optional icons.
/// Returns the rect the value should occupy.
fn shell(ui: &Ui, rect: Rect, response: &Response, label: &str, leading: Option<char>, active: bool) -> Rect {
    let colors = theme::colors(ui.ctx());
    let enabled = response.enabled();
    let painter = ui.painter();
    painter.rect_filled(rect, TOP_ROUNDED, paint::container(colors.surface_container_highest, enabled));
    paint::state_layer(ui, response, rect, TOP_ROUNDED, colors.on_surface);

    let focus = ui.ctx().animate_bool_with_time(response.id.with("active"), active, motion::SHORT_4);
    let indicator_color = colors.on_surface_variant.lerp_to_gamma(colors.primary, focus);
    let thickness = egui::lerp(INDICATOR..=INDICATOR_FOCUSED, focus);
    let indicator = Rect::from_x_y_ranges(rect.x_range(), rect.bottom() - thickness..=rect.bottom());
    painter.rect_filled(indicator, CornerRadius::ZERO, indicator_color);

    let mut left = rect.left() + PADDING_X;
    if let Some(icon) = leading {
        let center = pos2(left + size::ICON / 2.0, rect.center().y);
        paint::icon(painter, center, icon, size::ICON, colors.on_surface_variant, false);
        left += size::ICON + ICON_GAP;
    }
    let label_color = colors.on_surface_variant.lerp_to_gamma(colors.primary, focus);
    paint::text(ui, pos2(left, rect.top() + LABEL_TOP), Align2::LEFT_TOP, TypeRole::BodySmall, label, label_color);
    Rect::from_min_max(pos2(left, rect.top() + VALUE_TOP), pos2(rect.right() - PADDING_X, rect.bottom() - spacing::SM))
}

/// A single-line filled text field with a persistent (floating) label.
#[must_use = "add it with `ui.add(...)`"]
pub struct TextField<'a> {
    label: &'a str,
    text: &'a mut String,
    icon: Option<char>,
    hint: &'a str,
}

impl<'a> TextField<'a> {
    /// A field editing `text`.
    pub fn new(label: &'a str, text: &'a mut String) -> Self {
        Self { label, text, icon: None, hint: "" }
    }

    /// Leading icon.
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Placeholder shown while empty.
    pub fn hint(mut self, hint: &'a str) -> Self {
        self.hint = hint;
        self
    }
}

impl Widget for TextField<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let colors = theme::colors(ui.ctx());
        let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
        let edit_id = response.id.with("edit");
        let focused = ui.memory(|m| m.has_focus(edit_id));
        let value_rect = shell(ui, rect, &response, self.label, self.icon, focused);
        let text_color = colors.on_surface;
        let edit = egui::TextEdit::singleline(self.text)
            .id(edit_id)
            .frame(false)
            .margin(Margin::ZERO)
            .font(TypeRole::BodyLarge.text_style())
            .text_color(text_color)
            .hint_text(TypeRole::BodyLarge.rich(self.hint, colors.on_surface_variant))
            .desired_width(value_rect.width());
        ui.put(value_rect, edit)
    }
}

/// A filled exposed dropdown menu choosing one of `options`.
#[must_use = "add it with `ui.add(...)`"]
pub struct Dropdown<'a, T> {
    label: &'a str,
    value: &'a mut T,
    options: &'a [(T, &'a str)],
    icon: Option<char>,
}

impl<'a, T: PartialEq + Copy> Dropdown<'a, T> {
    /// A dropdown editing `value`.
    pub fn new(label: &'a str, value: &'a mut T, options: &'a [(T, &'a str)]) -> Self {
        Self { label, value, options, icon: None }
    }

    /// Leading icon.
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }
}

impl<T: PartialEq + Copy> Widget for Dropdown<'_, T> {
    fn ui(self, ui: &mut Ui) -> Response {
        let colors = theme::colors(ui.ctx());
        let enabled = ui.is_enabled();
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, mut response) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), sense);
        let response_cursor = if enabled { CursorIcon::PointingHand } else { CursorIcon::Default };
        response = response.on_hover_cursor(response_cursor);
        let selected_label = self.options.iter().find(|(v, _)| v == self.value).map_or("", |(_, l)| *l);
        response.widget_info(|| WidgetInfo::labeled(WidgetType::ComboBox, enabled, selected_label));

        let popup =
            Popup::menu(&response).width(rect.width()).gap(0.0).close_behavior(PopupCloseBehavior::CloseOnClick).frame(menu_frame(&colors));
        let open = popup.is_open();
        let value_rect = shell(ui, rect, &response, self.label, self.icon, open);
        let text_color = colors.on_surface;
        let value_width = value_rect.width() - size::ICON - ICON_GAP; // leave room for the arrow
        paint::text_truncated(
            ui,
            value_rect.left_center(),
            Align2::LEFT_CENTER,
            TypeRole::BodyLarge,
            selected_label,
            text_color,
            value_width,
        );
        let arrow = pos2(rect.right() - PADDING_X - size::ICON / 2.0, rect.center().y);
        let glyph = if open { icons::ARROW_DROP_UP } else { icons::ARROW_DROP_DOWN };
        paint::icon(ui.painter(), arrow, glyph, size::ICON, colors.on_surface_variant, false);

        let chosen = popup
            .show(|ui| {
                let mut chosen = None;
                for (option, label) in self.options {
                    if menu_item(ui, label, option == self.value).clicked() {
                        chosen = Some(*option);
                    }
                }
                chosen
            })
            .and_then(|inner| inner.inner);
        if let Some(option) = chosen
            && option != *self.value
        {
            *self.value = option;
            response.mark_changed();
        }
        response
    }
}

/// MD3 menu surface: `surfaceContainer`, extra-small corners, level-2 elevation.
pub fn menu_frame(colors: &ColorScheme) -> Frame {
    Frame::new()
        .fill(colors.surface_container)
        .corner_radius(shape::EXTRA_SMALL)
        .shadow(Elevation::Level2.shadow(colors.shadow))
        .inner_margin(Margin::symmetric(0, MENU_PADDING_Y))
}

/// A 48 dp menu row; the selected row gets a check mark and a tonal fill.
pub fn menu_item(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let colors = theme::colors(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), MENU_ITEM_HEIGHT), Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::SelectableLabel, true, selected, label));
    if selected {
        ui.painter().rect_filled(rect, CornerRadius::ZERO, colors.secondary_container);
    }
    let content = if selected { colors.on_secondary_container } else { colors.on_surface };
    paint::state_layer(ui, &response, rect, CornerRadius::ZERO, content);
    let mut x = rect.left() + spacing::MD;
    let icon_center = pos2(x + size::ICON / 2.0, rect.center().y);
    if selected {
        paint::icon(ui.painter(), icon_center, icons::CHECK, size::ICON, content, false);
    }
    x += size::ICON + spacing::MD;
    let width = rect.right() - x - spacing::MD;
    paint::text_truncated(ui, pos2(x, rect.center().y), Align2::LEFT_CENTER, TypeRole::LabelLarge, label, content, width);
    response
}
