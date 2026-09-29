//! Swap model and quality, face selection, paste mask and enhancement.

use super::super::Frame;
use super::{caption, description, labeled_slider, percent, percent_points, section, switch_row, toggle_section};
use crate::ui::components::{Dropdown, SegmentedButton};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, spacing};
use eframe::egui::{Align, Layout, Ui};
use wajh_core::assets::SwapModel;
use wajh_core::pipeline::FaceOrder;
use wajh_core::pipeline::pixel_boost::boosted_size;

// ── Swap ───────────────────────────────────────────────────────────────────────

const MODELS: [(SwapModel, &str); 4] = [
    (SwapModel::Inswapper, SwapModel::Inswapper.label()),
    (SwapModel::Hyperswap1a, SwapModel::Hyperswap1a.label()),
    (SwapModel::Hyperswap1b, SwapModel::Hyperswap1b.label()),
    (SwapModel::Hyperswap1c, SwapModel::Hyperswap1c.label()),
];
const FACE_ORDERS: [(FaceOrder, &str); 5] = [
    (FaceOrder::LargeToSmall, FaceOrder::LargeToSmall.label()),
    (FaceOrder::LeftToRight, FaceOrder::LeftToRight.label()),
    (FaceOrder::RightToLeft, FaceOrder::RightToLeft.label()),
    (FaceOrder::TopToBottom, FaceOrder::TopToBottom.label()),
    (FaceOrder::BestScore, FaceOrder::BestScore.label()),
];
const DETECTOR_SIZES: [(usize, &str); 3] = [(320, "320"), (480, "480"), (640, "640")];
/// Swap output sizes offered for pixel boost (only whole multiples of the model's native size apply).
const BOOST_SIZES: [usize; 4] = [128, 256, 384, 512];

pub(super) fn swap(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    section(ui, icons::SWAP_HORIZ, "Swap", |ui| {
        ui.add(Dropdown::new("Model", &mut settings.swap_model, &MODELS).icon(icons::MEMORY));
        pixel_boost(ui, settings.swap_model.resolution(), &mut settings.pixel_boost);
        switch_row(ui, icons::GROUPS, "All faces", "Swap every face, not just the first", &mut settings.many_faces);
        ui.add(Dropdown::new("Face order", &mut settings.face_order, &FACE_ORDERS).icon(icons::SORT));
        switch_row(ui, icons::FLIP, "Mirror", "Flip the picture like a selfie camera", &mut settings.mirror);
        switch_row(ui, icons::CENTER_FOCUS_STRONG, "Stabilise", "Smooth landmark jitter between frames", &mut settings.stabilize);
        caption(ui, "Detector size");
        ui.add(SegmentedButton::new(&mut settings.detector_size, &DETECTOR_SIZES));
    });
}

/// Output-size segments with the swap-time cost of the current choice.
/// `boost` stores `0` for the model's native size, as [`wajh_core::Settings`] expects.
fn pixel_boost(ui: &mut Ui, native: usize, boost: &mut usize) {
    let labels: Vec<(usize, String)> =
        BOOST_SIZES.iter().filter(|&&size| size % native == 0).map(|&size| (size, size.to_string())).collect();
    let options: Vec<(usize, &str)> = labels.iter().map(|(size, label)| (*size, label.as_str())).collect();
    let mut size = boosted_size(native, *boost);
    let factor = (size / native).pow(2);
    let cost = if factor == 1 { "native speed".to_owned() } else { format!("×{factor} slower") };
    let colors = theme::colors(ui.ctx());
    ui.add_space(spacing::XS);
    ui.horizontal(|ui| {
        ui.label(TypeRole::LabelLarge.rich("Pixel boost", colors.on_surface_variant));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let color = if factor == 1 { colors.on_surface_variant } else { colors.tertiary };
            ui.label(TypeRole::LabelLarge.rich(cost, color));
        });
    });
    if ui.add(SegmentedButton::new(&mut size, &options)).changed() {
        *boost = if size == native { 0 } else { size };
    }
}

// ── Mask ───────────────────────────────────────────────────────────────────────

const MAX_BLUR: f32 = 1.0;
const MAX_PADDING: f32 = 50.0;
const PADDING_SIDES: [&str; 4] = ["Top", "Right", "Bottom", "Left"];

pub(super) fn mask(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    section(ui, icons::BLUR_ON, "Mask", |ui| {
        labeled_slider(ui, "Edge blur", &mut settings.mask_blur, 0.0..=MAX_BLUR, percent);
        caption(ui, "Padding");
        ui.spacing_mut().item_spacing.x = spacing::XL;
        let [top, right, bottom, left] = &mut settings.mask_padding;
        for row in [[(PADDING_SIDES[0], top), (PADDING_SIDES[1], right)], [(PADDING_SIDES[2], bottom), (PADDING_SIDES[3], left)]] {
            ui.columns(2, |columns| {
                for (column, (side, value)) in columns.iter_mut().zip(row) {
                    labeled_slider(column, side, value, 0.0..=MAX_PADDING, percent_points);
                }
            });
        }
    });
}

// ── Enhance ────────────────────────────────────────────────────────────────────

pub(super) fn enhance(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    toggle_section(ui, icons::AUTO_AWESOME, "Enhance", &mut settings.enhance, |ui| {
        description(ui, "Restore skin and eye detail with GPEN. Costs a few milliseconds per face.");
        labeled_slider(ui, "Blend", &mut settings.enhance_blend, 0.0..=1.0, percent);
    });
}
