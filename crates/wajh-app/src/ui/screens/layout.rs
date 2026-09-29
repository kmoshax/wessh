//! Layout helpers shared by the screens.

use crate::ui::components::{Chip, TextField};
use crate::ui::icons;
use crate::ui::theme::{self, shape, spacing};
use eframe::egui::{Align, Id, Key, Layout, Rect, Ui, UiBuilder, pos2};
use wajh_core::Stats;

/// Place content of unknown height centred in `rect`, using last frame's measured height.
pub fn centered(ui: &mut Ui, rect: Rect, id: &str, add: impl FnOnce(&mut Ui)) {
    let id = Id::new(id);
    let height: f32 = ui.ctx().data(|d| d.get_temp(id)).unwrap_or_default();
    let top = rect.center().y - height / 2.0;
    let child = Rect::from_min_max(pos2(rect.left(), top.max(rect.top())), rect.max);
    let used = ui.scope_builder(UiBuilder::new().max_rect(child).layout(Layout::top_down(Align::Center)), add).response.rect;
    if (used.height() - height).abs() > 0.5 {
        ui.ctx().data_mut(|d| d.insert_temp(id, used.height()));
        ui.ctx().request_repaint();
    }
}

/// The rounded `surfaceContainerLowest` pane every screen draws its main content on.
pub fn pane(ui: &Ui, rect: Rect) {
    let colors = theme::colors(ui.ctx());
    ui.painter().rect_filled(rect, shape::EXTRA_LARGE, colors.surface_container_lowest);
}

/// A column of at most `max_width`, centred horizontally in `rect`.
#[must_use]
pub fn column(rect: Rect, max_width: f32) -> Rect {
    let width = max_width.min(rect.width() - 2.0 * spacing::XL).max(0.0);
    Rect::from_center_size(rect.center(), eframe::egui::vec2(width, rect.height()))
}

/// Path entry for systems with neither a file dialog nor drag-and-drop. Returns the path once
/// submitted (Enter or the Open button).
pub fn path_field(ui: &mut Ui, label: &str, hint: &str, text: &mut String) -> Option<std::path::PathBuf> {
    let submitted =
        ui.add(TextField::new(label, text).icon(icons::FOLDER_OPEN).hint(hint)).lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
    ui.add_space(spacing::SM);
    let clicked = ui.add_enabled(!text.trim().is_empty(), crate::ui::components::Button::tonal("Open").icon(icons::UPLOAD)).clicked();
    (submitted || clicked).then(|| std::path::PathBuf::from(text.trim()))
}

/// Faces and per-stage timings as status chips, laid out in the current direction.
pub fn stat_chips(ui: &mut Ui, stats: &Stats) {
    ui.spacing_mut().item_spacing.x = spacing::SM;
    let faces = if stats.faces == 1 { "1 face".to_owned() } else { format!("{} faces", stats.faces) };
    ui.add(Chip::status(&faces, false).icon(icons::FACE));
    ui.add(Chip::status(&format!("{:.0} ms", stats.total_ms), true).icon(icons::TIMER));
    let stages = [("Detect", stats.detect_ms), ("Swap", stats.swap_ms), ("Mask", stats.mask_ms), ("Enhance", stats.enhance_ms)];
    for (stage, ms) in stages.into_iter().filter(|(_, ms)| *ms > 0.0) {
        ui.add(Chip::status(&format!("{stage} {ms:.1} ms"), false));
    }
}
