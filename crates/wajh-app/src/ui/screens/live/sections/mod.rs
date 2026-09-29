//! Settings side-sheet sections, one small function each.
//!
//! To add a section, write a `fn(&mut Ui, &mut Frame)` that wraps its content in
//! [`section`] (or [`toggle_section`]) and list it in [`ALL`].

mod input;
mod quality;
mod swap;

use super::Frame;
use crate::ui::components::{Card, ListItem, Slider, Switch, paint};
use crate::ui::theme::{self, TypeRole, size, spacing};
use eframe::egui::{Align, Label, Layout, Sense, Ui, Vec2, vec2};

/// A settings section.
pub type Section = fn(&mut Ui, &mut Frame<'_>);

/// Sections in display order.
pub const ALL: [Section; 10] = [
    input::source,
    input::camera,
    input::output,
    swap::swap,
    swap::mask,
    quality::occlusion,
    quality::regions,
    quality::mouth,
    quality::color,
    swap::enhance,
];

const HEADER_HEIGHT: f32 = 40.0;

/// A card with an icon + title header and `body` below.
fn section(ui: &mut Ui, icon: char, title: &str, body: impl FnOnce(&mut Ui)) {
    Card::elevated().show(ui, |ui| {
        header(ui, icon, title, |_| {});
        body(ui);
    });
}

/// A section whose header carries an on/off switch; the body is dimmed while off.
fn toggle_section(ui: &mut Ui, icon: char, title: &str, on: &mut bool, body: impl FnOnce(&mut Ui)) {
    Card::elevated().show(ui, |ui| {
        header(ui, icon, title, |ui| {
            ui.add(Switch::new(on));
        });
        let enabled = *on;
        ui.add_enabled_ui(enabled, body);
    });
}

fn header(ui: &mut Ui, icon: char, title: &str, trailing: impl FnOnce(&mut Ui)) {
    let colors = theme::colors(ui.ctx());
    ui.allocate_ui_with_layout(vec2(ui.available_width(), HEADER_HEIGHT), Layout::left_to_right(Align::Center), |ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON), Sense::hover());
        paint::icon(ui.painter(), rect.center(), icon, size::ICON, colors.primary, false);
        ui.add_space(spacing::XS);
        ui.label(TypeRole::TitleMedium.rich(title, colors.on_surface));
        ui.with_layout(Layout::right_to_left(Align::Center), trailing);
    });
}

/// A small caption above a control group.
fn caption(ui: &mut Ui, text: &str) {
    let colors = theme::colors(ui.ctx());
    ui.add_space(spacing::XS);
    ui.label(TypeRole::LabelLarge.rich(text, colors.on_surface_variant));
}

/// A caption with the formatted value on the right, followed by a slider.
fn labeled_slider(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, format: fn(f32) -> String) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        ui.label(TypeRole::BodyLarge.rich(label, colors.on_surface));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(TypeRole::LabelLarge.rich(format(*value), colors.on_surface_variant));
        });
    });
    ui.add(Slider::new(value, range).format(format));
}

/// A list row with a trailing switch; clicking anywhere on the row toggles it.
fn switch_row(ui: &mut Ui, icon: char, headline: &str, supporting: &str, on: &mut bool) {
    let (row, ()) = ListItem::new(headline).supporting(supporting).icon(icon).show(ui, |ui| {
        ui.add(Switch::new(on));
    });
    if row.clicked() {
        *on = !*on;
    }
}

fn percent(value: f32) -> String {
    format!("{:.0} %", value * 100.0)
}

fn percent_points(value: f32) -> String {
    format!("{value:.0} %")
}

/// Supporting text under a section header, wrapped to the card width.
fn description(ui: &mut Ui, text: &str) {
    let colors = theme::colors(ui.ctx());
    let color = colors.on_surface_variant;
    let galley = ui.fonts_mut(|f| f.layout_job(TypeRole::BodyMedium.job(text, color, Some(ui.available_width()))));
    ui.add(Label::new(galley).selectable(false));
}
