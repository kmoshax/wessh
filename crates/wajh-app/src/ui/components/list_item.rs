//! MD3 list items: leading icon, headline, supporting text and a trailing control.

use super::paint;
use crate::ui::theme::{self, TypeRole, shape, size, spacing};
use eframe::egui::{Align, Layout, Rect, Response, Sense, Ui, UiBuilder, pos2, vec2};

const ONE_LINE_HEIGHT: f32 = 56.0;
const TWO_LINE_HEIGHT: f32 = 72.0;
const VERTICAL_PADDING: f32 = spacing::SM;
const LEADING_GAP: f32 = spacing::LG;
/// Room reserved for trailing controls such as a switch.
const TRAILING_WIDTH: f32 = 60.0;

/// One row of a list.
#[must_use = "call `.show(ui, ...)`"]
pub struct ListItem<'a> {
    headline: &'a str,
    supporting: Option<&'a str>,
    icon: Option<char>,
}

impl<'a> ListItem<'a> {
    /// A row with a headline.
    pub fn new(headline: &'a str) -> Self {
        Self { headline, supporting: None, icon: None }
    }

    /// Secondary line under the headline.
    pub fn supporting(mut self, text: &'a str) -> Self {
        self.supporting = Some(text);
        self
    }

    /// Leading icon.
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Show the row; `trailing` fills the right edge. Returns the row's own response
    /// (click anywhere on the row) together with the trailing closure's result.
    pub fn show<R>(self, ui: &mut Ui, trailing: impl FnOnce(&mut Ui) -> R) -> (Response, R) {
        let colors = theme::colors(ui.ctx());
        let enabled = ui.is_enabled();
        let width = ui.available_width();
        let text_left = if self.icon.is_some() { size::ICON + LEADING_GAP } else { 0.0 };
        let text_width = (width - text_left - TRAILING_WIDTH).max(0.0);
        let headline_color = paint::content(colors.on_surface, enabled);
        let supporting_color = paint::content(colors.on_surface_variant, enabled);
        let headline = ui.fonts_mut(|f| f.layout_job(TypeRole::BodyLarge.job(self.headline, headline_color, Some(text_width))));
        let supporting =
            self.supporting.map(|text| ui.fonts_mut(|f| f.layout_job(TypeRole::BodyMedium.job(text, supporting_color, Some(text_width)))));
        let text_height = headline.size().y + supporting.as_ref().map_or(0.0, |g| g.size().y);
        let min_height = if supporting.is_some() { TWO_LINE_HEIGHT } else { ONE_LINE_HEIGHT };
        let height = min_height.max(text_height + 2.0 * VERTICAL_PADDING);

        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, response) = ui.allocate_exact_size(vec2(width, height), sense);
        paint::state_layer(ui, &response, rect, shape::SMALL, colors.on_surface);

        if let Some(icon) = self.icon {
            let center = pos2(rect.left() + size::ICON / 2.0, rect.center().y);
            paint::icon(ui.painter(), center, icon, size::ICON, supporting_color, false);
        }
        let mut y = rect.center().y - text_height / 2.0;
        let x = rect.left() + text_left;
        ui.painter().galley(pos2(x, y), headline.clone(), headline_color);
        y += headline.size().y;
        if let Some(galley) = supporting {
            ui.painter().galley(pos2(x, y), galley, supporting_color);
        }

        let trailing_rect = Rect::from_x_y_ranges(rect.right() - TRAILING_WIDTH..=rect.right(), rect.y_range());
        let inner = ui.scope_builder(UiBuilder::new().max_rect(trailing_rect).layout(Layout::right_to_left(Align::Center)), trailing);
        (response, inner.inner)
    }
}
