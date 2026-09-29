//! Image and Video screens. File processing is not wired into the app yet, so these
//! explain what is coming and show the equivalent command-line invocation.

use super::Screen;
use crate::ui::components::{Card, empty_state};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, shape, spacing};
use eframe::egui::{Align, Layout, Rect, Ui, UiBuilder, vec2};

const CONTENT_WIDTH: f32 = 520.0;

struct Blurb {
    icon: char,
    title: &'static str,
    body: &'static str,
    command: &'static str,
}

const IMAGE: Blurb = Blurb {
    icon: icons::IMAGE,
    title: "Swap faces in a photo",
    body: "Pick a source face and a target photo, preview the result and save it. Coming to the app soon — the command line already does it:",
    command: "wajh image -s face.jpg -t photo.jpg -o out.jpg",
};

const VIDEO: Blurb = Blurb {
    icon: icons::MOVIE,
    title: "Swap faces in a video",
    body: "Render a whole clip with hardware encoding and the original audio. Coming to the app soon — the command line already does it:",
    command: "wajh video -s face.jpg -t clip.mp4 -o out.mp4",
};

/// Draw the Image or Video screen into `rect`.
pub fn offline_screen(ui: &mut Ui, rect: Rect, screen: Screen) {
    let colors = theme::colors(ui.ctx());
    let copy = if screen == Screen::Video { &VIDEO } else { &IMAGE };
    ui.painter().rect_filled(rect, shape::EXTRA_LARGE, colors.surface_container_lowest);
    let width = CONTENT_WIDTH.min(rect.width() - 2.0 * spacing::XL);
    let column = Rect::from_center_size(rect.center(), vec2(width, rect.height()));
    ui.scope_builder(UiBuilder::new().max_rect(column).layout(Layout::top_down(Align::Center)), |ui| {
        ui.add_space((rect.height() / 4.0).max(spacing::XL));
        empty_state(ui, copy.icon, copy.title, copy.body, |ui| {
            Card::outlined().padding(spacing::MD).show(ui, |ui| {
                ui.label(TypeRole::BodyMedium.rich(copy.command, colors.on_surface).monospace());
            });
        });
    });
}
