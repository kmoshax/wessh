//! Swap quality, face selection, paste mask and enhancement. The swap model itself is not a
//! choice: wessh ships one curated model (see [`wessh_core::assets::required`]).

use super::super::Frame;
use super::{caption, description, labeled_slider, percent, percent_points, section, switch_row, toggle_section};
use crate::ui::components::{Dropdown, SegmentedButton};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, spacing};
use eframe::egui::{Align, Layout, Ui};
use wessh_core::pipeline::FaceOrder;
use wessh_core::pipeline::pixel_boost::{AUTO, AUTO_SIZE, boosted_size};

// ── Swap ───────────────────────────────────────────────────────────────────────

const FACE_ORDERS: [(FaceOrder, &str); 5] = [
    (FaceOrder::LargeToSmall, FaceOrder::LargeToSmall.label()),
    (FaceOrder::LeftToRight, FaceOrder::LeftToRight.label()),
    (FaceOrder::RightToLeft, FaceOrder::RightToLeft.label()),
    (FaceOrder::TopToBottom, FaceOrder::TopToBottom.label()),
    (FaceOrder::BestScore, FaceOrder::BestScore.label()),
];
const DETECTOR_SIZES: [(usize, &str); 3] = [(320, "320"), (480, "480"), (640, "640")];
/// Largest pixel-boost output offered: `FaceFusion`'s largest (512 px); beyond it the cost
/// ((size / native)² swaps per face) outgrows the gain.
const MAX_BOOST_SIZE: usize = 512;

pub(super) fn swap(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    let live = !frame.offline;
    section(ui, icons::SWAP_HORIZ, "Swap", |ui| {
        // The curated model (`SwapModel::default()`, unless a developer picked another with --model).
        pixel_boost(ui, settings.swap_model.resolution(), &mut settings.pixel_boost);
        switch_row(ui, icons::GROUPS, "All faces", "Swap every face, not just the first", &mut settings.many_faces);
        ui.add(Dropdown::new("Face order", &mut settings.face_order, &FACE_ORDERS).icon(icons::SORT));
        if live {
            switch_row(ui, icons::FLIP, "Mirror", "Flip the picture like a selfie camera", &mut settings.mirror);
            switch_row(ui, icons::CENTER_FOCUS_STRONG, "Stabilise", "Smooth landmark jitter between frames", &mut settings.stabilize);
        }
        caption(ui, "Detector size");
        ui.add(SegmentedButton::new(&mut settings.detector_size, &DETECTOR_SIZES));
    });
}

/// Output-size segments (whole multiples of the model's `native` resolution up to
/// [`MAX_BOOST_SIZE`]) with the swap-time cost of the current choice.
/// `boost` stores [`AUTO`] or the size, as [`wessh_core::Settings`] expects.
fn pixel_boost(ui: &mut Ui, native: usize, boost: &mut usize) {
    let mut labels = vec![(AUTO, "Auto".to_owned())];
    labels.extend(boost_sizes(native).map(|size| (size, size.to_string())));
    let options: Vec<(usize, &str)> = labels.iter().map(|(size, label)| (*size, label.as_str())).collect();
    let mut size = if *boost == AUTO { AUTO } else { boosted_size(native, *boost) };
    let factor = (boosted_size(native, size) / native).pow(2);
    let cost = if size == AUTO && native < AUTO_SIZE {
        format!("{AUTO_SIZE} for close faces")
    } else if factor == 1 {
        "native speed".to_owned()
    } else {
        format!("×{factor} slower")
    };
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
        *boost = size;
    }
}

/// Pixel-boost sizes for a model of `native` resolution: ×1, ×2, … up to [`MAX_BOOST_SIZE`]
/// (always at least the native size).
fn boost_sizes(native: usize) -> impl Iterator<Item = usize> {
    (1..).map(move |factor| factor * native).take_while(move |&size| size == native || size <= MAX_BOOST_SIZE)
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
        description(ui, "Restore skin and eye detail with GFPGAN, as much as the camera picture itself holds.");
        labeled_slider(ui, "Blend", &mut settings.enhance_blend, 0.0..=1.0, percent);
    });
}

#[cfg(test)]
mod tests {
    use super::boost_sizes;

    #[test]
    fn boost_sizes_are_multiples_of_the_native_resolution() {
        assert_eq!(boost_sizes(128).collect::<Vec<_>>(), [128, 256, 384, 512]);
        assert_eq!(boost_sizes(256).collect::<Vec<_>>(), [256, 512]);
        assert_eq!(boost_sizes(512).collect::<Vec<_>>(), [512]);
        assert_eq!(boost_sizes(1024).collect::<Vec<_>>(), [1024]);
    }
}
