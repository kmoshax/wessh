//! MD3 top app bar and navigation rail.

use super::paint;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{Align, Align2, CursorIcon, Layout, Rect, Response, Sense, Ui, UiBuilder, Vec2, WidgetInfo, WidgetType, pos2, vec2};

/// Height of the small top app bar.
pub const TOP_APP_BAR_HEIGHT: f32 = 64.0;
/// Width of the navigation rail.
pub const RAIL_WIDTH: f32 = 80.0;

const RAIL_INDICATOR: Vec2 = Vec2::new(56.0, 32.0);
const RAIL_ITEM_HEIGHT: f32 = 56.0;
const RAIL_ITEM_GAP: f32 = spacing::MD;
const RAIL_LABEL_GAP: f32 = spacing::XS;
const LOGO_SIZE: f32 = 32.0;

/// Paint the small top app bar content: logo mark, title and trailing actions.
pub fn top_app_bar(ui: &mut Ui, title: &str, actions: impl FnOnce(&mut Ui)) {
    let colors = theme::colors(ui.ctx());
    let rect = ui.max_rect();
    let logo_rect = Rect::from_center_size(pos2(rect.left() + RAIL_WIDTH / 2.0, rect.center().y), Vec2::splat(LOGO_SIZE));
    ui.painter().rect_filled(logo_rect, shape::SMALL, colors.primary_container);
    let logo_id = eframe::egui::Id::new("happy-face-logo");
    let texture = ui.ctx().data_mut(|data| {
        data.get_temp_mut_or_insert_with::<eframe::egui::TextureHandle>(logo_id, || {
            let image = image::load_from_memory(include_bytes!("../../../assets/icon/happy-face.png"))
                .expect("the embedded logo is a valid PNG")
                .into_rgba8();
            let pixels =
                eframe::egui::ColorImage::from_rgba_unmultiplied([image.width() as usize, image.height() as usize], image.as_raw());
            ui.ctx().load_texture("happy-face-logo", pixels, eframe::egui::TextureOptions::LINEAR)
        })
        .clone()
    });
    ui.painter().image(
        texture.id(),
        Rect::from_center_size(logo_rect.center(), Vec2::splat(size::ICON)),
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        colors.on_primary_container,
    );
    paint::text(ui, pos2(rect.left() + RAIL_WIDTH, rect.center().y), Align2::LEFT_CENTER, TypeRole::TitleLarge, title, colors.on_surface);

    let actions_rect = Rect::from_x_y_ranges(rect.center().x..=rect.right() - spacing::SM, rect.y_range());
    ui.scope_builder(UiBuilder::new().max_rect(actions_rect).layout(Layout::right_to_left(Align::Center)), actions);
}

/// One destination on the navigation rail.
#[derive(Clone, Copy, Debug)]
pub struct Destination<'a, T> {
    pub value: T,
    pub icon: char,
    pub label: &'a str,
}

/// A vertical navigation rail. Returns `true` when the selection changed.
pub fn navigation_rail<T: PartialEq + Copy>(ui: &mut Ui, selected: &mut T, destinations: &[Destination<'_, T>]) -> bool {
    let mut changed = false;
    ui.add_space(spacing::XS);
    for destination in destinations {
        let active = *selected == destination.value;
        if rail_item(ui, destination, active).clicked() && !active {
            *selected = destination.value;
            changed = true;
        }
        ui.add_space(RAIL_ITEM_GAP);
    }
    changed
}

fn rail_item<T>(ui: &mut Ui, destination: &Destination<'_, T>, active: bool) -> Response {
    let colors = theme::colors(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(vec2(RAIL_WIDTH, RAIL_ITEM_HEIGHT), Sense::click());
    let response = response.on_hover_cursor(CursorIcon::PointingHand);
    response.widget_info(|| WidgetInfo::selected(WidgetType::SelectableLabel, true, active, destination.label));

    let indicator_center = pos2(rect.center().x, rect.top() + RAIL_INDICATOR.y / 2.0);
    let t = ui.ctx().animate_bool_with_time_and_easing(response.id, active, motion::MEDIUM_2, motion::emphasized);
    let indicator_full = Rect::from_center_size(indicator_center, RAIL_INDICATOR);
    let radius = shape::full(RAIL_INDICATOR.y);
    if t > 0.0 {
        let indicator = Rect::from_center_size(indicator_center, vec2(RAIL_INDICATOR.x * t, RAIL_INDICATOR.y));
        ui.painter().rect_filled(indicator, radius, colors.secondary_container);
    }
    let icon_color = colors.on_surface_variant.lerp_to_gamma(colors.on_secondary_container, t);
    paint::state_layer(ui, &response, indicator_full, radius, colors.on_surface);
    paint::focus_ring(ui, &response, indicator_full, radius, &colors);
    paint::icon(ui.painter(), indicator_center, destination.icon, size::ICON, icon_color, active);

    let label_color = if active { colors.on_surface } else { colors.on_surface_variant };
    let label_pos = pos2(rect.center().x, indicator_full.bottom() + RAIL_LABEL_GAP);
    paint::text(ui, label_pos, Align2::CENTER_TOP, TypeRole::LabelMedium, destination.label, label_color);
    response
}
