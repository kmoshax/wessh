//! Feedback components: circular progress indicator, snackbar and empty states.

use super::button::IconButton;
use super::paint;
use crate::ui::icons;
use crate::ui::theme::{self, Elevation, TypeRole, motion, shape, size, spacing};
use eframe::egui::{
    Align2, Area, Context, Id, LayerId, Order, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2, Widget,
    epaint::PathStroke, pos2, vec2,
};
use std::f32::consts::TAU;

/// Default diameter of the circular indicator.
pub const PROGRESS_SIZE: f32 = 48.0;
const PROGRESS_STROKE: f32 = 4.0;
/// Seconds for one grow/shrink cycle of the indeterminate arc.
const PROGRESS_CYCLE: f64 = 1.333;
/// Full turns the whole indicator makes per cycle.
const PROGRESS_SPIN: f64 = 0.8;
const ARC_MIN: f32 = 0.03;
const ARC_MAX: f32 = 0.75;
const ARC_SEGMENTS: usize = 48;

/// MD3 indeterminate circular progress indicator.
#[must_use = "add it with `ui.add(...)`"]
pub struct CircularProgress {
    size: f32,
}

impl CircularProgress {
    /// An indicator of the given diameter.
    pub fn new(size: f32) -> Self {
        Self { size }
    }
}

impl Widget for CircularProgress {
    fn ui(self, ui: &mut Ui) -> Response {
        let colors = theme::colors(ui.ctx());
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(self.size), Sense::hover());
        let time = ui.input(|i| i.time);
        let phase = (time / PROGRESS_CYCLE).fract() as f32;
        // The head leads during the first half of the cycle, the tail catches up in the second.
        let grow = motion::standard((phase * 2.0).min(1.0));
        let shrink = motion::standard((phase * 2.0 - 1.0).max(0.0));
        let spin = ((time * PROGRESS_SPIN / PROGRESS_CYCLE).fract() as f32 + (time / PROGRESS_CYCLE).floor() as f32 * ARC_MAX) * TAU;
        let head = spin + (ARC_MIN + (ARC_MAX - ARC_MIN) * grow) * TAU;
        let tail = spin + (ARC_MAX - ARC_MIN) * shrink * TAU;
        let radius = (self.size - PROGRESS_STROKE) / 2.0;
        let points = arc_points(rect.center(), radius, tail, head);
        ui.painter().add(Shape::line(points, PathStroke::new(PROGRESS_STROKE, colors.primary)));
        ui.ctx().request_repaint();
        response
    }
}

fn arc_points(center: Pos2, radius: f32, from: f32, to: f32) -> Vec<Pos2> {
    (0..=ARC_SEGMENTS)
        .map(|i| {
            let angle = from + (to - from) * i as f32 / ARC_SEGMENTS as f32;
            center + radius * Vec2::angled(angle)
        })
        .collect()
}

/// Seconds a snackbar stays up.
const SNACKBAR_DURATION: f64 = 5.0;
const SNACKBAR_MIN_HEIGHT: f32 = 48.0;
const SNACKBAR_MAX_WIDTH: f32 = 560.0;
const SNACKBAR_MARGIN: f32 = spacing::LG;
const SNACKBAR_SLIDE: f32 = spacing::LG;

/// A transient message at the bottom of a region (MD3 snackbar).
#[derive(Default)]
pub struct Snackbar {
    message: Option<String>,
    shown_at: f64,
}

impl Snackbar {
    /// Queue `message`, replacing whatever is showing.
    pub fn show(&mut self, message: String, now: f64) {
        self.message = Some(message);
        self.shown_at = now;
    }

    /// Draw the snackbar anchored to the bottom of `region`.
    pub fn ui(&mut self, ui: &Ui, region: Rect) {
        let ctx = ui.ctx();
        let now = ctx.input(|i| i.time);
        let visible = self.message.is_some() && now - self.shown_at < SNACKBAR_DURATION;
        let id = Id::new("snackbar");
        let t = ctx.animate_bool_with_time_and_easing(id, visible, motion::MEDIUM_2, motion::emphasized);
        if t <= 0.0 {
            self.message = None;
            return;
        }
        if visible {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(SNACKBAR_DURATION - (now - self.shown_at)));
        }
        let Some(message) = self.message.clone() else { return };
        let colors = theme::colors(ctx);
        let width = (region.width() - 2.0 * SNACKBAR_MARGIN).min(SNACKBAR_MAX_WIDTH);
        let bottom = region.bottom() - SNACKBAR_MARGIN + SNACKBAR_SLIDE * (1.0 - t);
        let mut dismissed = false;
        Area::new(id).order(Order::Foreground).pivot(Align2::CENTER_BOTTOM).fixed_pos(pos2(region.center().x, bottom)).show(ctx, |ui| {
            ui.set_opacity(t);
            let frame = eframe::egui::Frame::new()
                .fill(colors.inverse_surface)
                .corner_radius(shape::EXTRA_SMALL)
                .shadow(Elevation::Level3.shadow(colors.shadow))
                .inner_margin(eframe::egui::Margin { left: spacing::LG as i8, right: spacing::XS as i8, top: 0, bottom: 0 });
            frame.show(ui, |ui| {
                ui.set_width(width - spacing::LG - spacing::XS);
                ui.set_min_height(SNACKBAR_MIN_HEIGHT);
                ui.horizontal_centered(|ui| {
                    let text_width = ui.available_width() - size::TARGET;
                    let job = TypeRole::BodyMedium.job(&message, colors.inverse_on_surface, Some(text_width));
                    let galley = ui.fonts_mut(|f| f.layout_job(job));
                    ui.add_sized(vec2(text_width, galley.size().y), eframe::egui::Label::new(galley).selectable(false));
                    let close = IconButton::new(icons::CLOSE, "Dismiss").inverse();
                    dismissed = ui.add(close).clicked();
                });
            });
        });
        if dismissed {
            self.shown_at = f64::NEG_INFINITY;
        }
    }
}

/// Size of the tonal circle behind an empty-state icon.
const EMPTY_ICON_CIRCLE: f32 = 80.0;
const EMPTY_ICON: f32 = 40.0;
const EMPTY_TEXT_WIDTH: f32 = 320.0;

/// A centred empty state: icon in a tonal circle, title, body and an optional action.
pub fn empty_state(ui: &mut Ui, icon: char, title: &str, body: &str, action: impl FnOnce(&mut Ui)) {
    let colors = theme::colors(ui.ctx());
    ui.vertical_centered(|ui| {
        let (circle, _) = ui.allocate_exact_size(Vec2::splat(EMPTY_ICON_CIRCLE), Sense::hover());
        ui.painter().circle_filled(circle.center(), EMPTY_ICON_CIRCLE / 2.0, colors.secondary_container);
        paint::icon(ui.painter(), circle.center(), icon, EMPTY_ICON, colors.on_secondary_container, false);
        ui.add_space(spacing::LG);
        ui.label(TypeRole::TitleLarge.rich(title, colors.on_surface));
        ui.add_space(spacing::XS);
        let wrap = EMPTY_TEXT_WIDTH.min(ui.available_width());
        let galley = ui.fonts_mut(|f| {
            let mut job = TypeRole::BodyMedium.job(body, colors.on_surface_variant, Some(wrap));
            job.halign = eframe::egui::Align::Center;
            f.layout_job(job)
        });
        ui.add(eframe::egui::Label::new(galley).selectable(false));
        ui.add_space(spacing::LG);
        action(ui);
    });
}

/// MD3 scrim opacity.
const SCRIM_OPACITY: f32 = 0.32;
const DROP_ZONE_OPACITY: f32 = 0.94;
const DROP_ICON: f32 = 64.0;

/// Full-window overlay shown while files are dragged over the app.
pub fn drop_overlay(ctx: &Context, rect: Rect, title: &str, body: &str, visible: bool) {
    let id = Id::new("drop-overlay");
    let t = ctx.animate_bool_with_time(id, visible, motion::SHORT_4);
    if t <= 0.0 {
        return;
    }
    let colors = theme::colors(ctx);
    let painter = ctx.layer_painter(LayerId::new(Order::Foreground, id));
    painter.rect_filled(rect, 0.0, colors.scrim.gamma_multiply(SCRIM_OPACITY * t));
    let zone = rect.shrink(spacing::XL);
    painter.rect_filled(zone, shape::EXTRA_LARGE, colors.primary_container.gamma_multiply(DROP_ZONE_OPACITY * t));
    let outline = Stroke::new(size::FOCUS_STROKE, colors.primary.gamma_multiply(t));
    painter.rect_stroke(zone, shape::EXTRA_LARGE, outline, StrokeKind::Inside);

    let content = colors.on_primary_container.gamma_multiply(t);
    let icon_center = zone.center() - vec2(0.0, DROP_ICON / 2.0 + spacing::LG);
    paint::icon(&painter, icon_center, icons::ADD_PHOTO_ALTERNATE, DROP_ICON, content, false);
    let mut y = zone.center().y;
    for (role, text) in [(TypeRole::HeadlineSmall, title), (TypeRole::BodyLarge, body)] {
        let galley = ctx.fonts_mut(|f| f.layout_job(role.job(text, content, None)));
        let size = galley.size();
        painter.galley(Align2::CENTER_TOP.anchor_size(pos2(zone.center().x, y), size).min, galley, content);
        y += size.y + spacing::XS;
    }
}
