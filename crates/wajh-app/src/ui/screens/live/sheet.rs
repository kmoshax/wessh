//! The settings side sheet: docked beside the preview on wide windows, modal on narrow ones.

use super::Frame;
use super::sections::Section;
use crate::ui::components::IconButton;
use crate::ui::icons;
use crate::ui::theme::{self, Elevation, TypeRole, motion, spacing};
use eframe::egui::{Align, Area, Context, CornerRadius, Id, Layout, Margin, Order, Rect, ScrollArea, Sense, Ui, UiBuilder, pos2, vec2};

/// Width of the side sheet (MD3 recommends up to 400 dp).
pub const SHEET_WIDTH: f32 = 360.0;
/// Gap between the preview pane and the docked sheet.
const PANE_GAP: f32 = spacing::XL;
/// MD3 scrim opacity behind modal sheets.
const SCRIM_OPACITY: f32 = 0.32;
const MODAL_HEADER_HEIGHT: f32 = 64.0;

/// Split the content area into preview (left) and sheet (right) rects.
#[must_use]
pub fn split(content: Rect) -> (Rect, Rect) {
    let sheet_left = content.right() - SHEET_WIDTH;
    let preview = Rect::from_min_max(content.min, pos2(sheet_left - PANE_GAP, content.bottom()));
    let sheet = Rect::from_min_max(pos2(sheet_left, content.top()), content.max);
    (preview, sheet)
}

/// Draw the sheet docked in `rect`.
pub fn docked(ui: &mut Ui, rect: Rect, frame: &mut Frame<'_>, sections: &[Section]) {
    ui.scope_builder(UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), |ui| {
        scroll_sections(ui, "docked-sheet", frame, sections);
    });
}

/// Draw the sheet as a modal from the window's right edge, sliding in and out as
/// `open` changes. Clicking the scrim or the close button clears `open`.
pub fn modal(ctx: &Context, frame: &mut Frame<'_>, open: &mut bool, sections: &[Section]) {
    let colors = theme::colors(ctx);
    let id = Id::new("modal-sheet");
    let t = ctx.animate_bool_with_time_and_easing(id, *open, motion::MEDIUM_4, motion::emphasized);
    if t <= 0.0 {
        return;
    }

    let screen = ctx.content_rect();
    Area::new(id.with("scrim")).order(Order::Middle).fixed_pos(screen.min).show(ctx, |ui| {
        let (rect, response) = ui.allocate_exact_size(screen.size(), Sense::click());
        ui.painter().rect_filled(rect, 0.0, colors.scrim.gamma_multiply(SCRIM_OPACITY * t));
        *open &= !response.clicked();
    });

    let width = SHEET_WIDTH.min(screen.width());
    let left = screen.right() - width * t;
    let sheet_rect = Rect::from_min_size(pos2(left, screen.top()), vec2(width, screen.height()));
    Area::new(id).order(Order::Foreground).fixed_pos(sheet_rect.min).show(ctx, |ui| {
        let radius = CornerRadius { nw: theme::shape::LARGE.nw, sw: theme::shape::LARGE.sw, ne: 0, se: 0 };
        eframe::egui::Frame::new()
            .fill(colors.surface_container_low)
            .corner_radius(radius)
            .shadow(Elevation::Level1.shadow(colors.shadow))
            .inner_margin(Margin { left: spacing::LG as i8, right: spacing::LG as i8, top: 0, bottom: 0 })
            .show(ui, |ui| {
                ui.set_width(width - 2.0 * spacing::LG);
                ui.set_height(screen.height());
                ui.allocate_ui_with_layout(vec2(ui.available_width(), MODAL_HEADER_HEIGHT), Layout::left_to_right(Align::Center), |ui| {
                    ui.label(TypeRole::TitleLarge.rich("Settings", colors.on_surface_variant));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        *open &= !ui.add(IconButton::new(icons::CLOSE, "Close")).clicked();
                    });
                });
                scroll_sections(ui, "modal-sheet", frame, sections);
            });
    });
}

fn scroll_sections(ui: &mut Ui, id: &str, frame: &mut Frame<'_>, sections: &[Section]) {
    ScrollArea::vertical().id_salt(id).auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = spacing::MD;
        for section in sections {
            section(ui, frame);
        }
        ui.add_space(spacing::LG);
    });
}
