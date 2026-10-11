//! Model-based masks and colour matching that make the swap blend in.

use super::super::Frame;
use super::{description, labeled_slider, percent, section, toggle_section};
use crate::ui::components::Chip;
use crate::ui::icons;
use crate::ui::theme::spacing;
use eframe::egui::Ui;
use weshy_core::face::FaceRegion;

/// Keep hands, hair and glasses in front of the face, with the curated occluder model.
pub(super) fn occlusion(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    toggle_section(ui, icons::BACK_HAND, "Occlusion", &mut settings.occlusion, |ui| {
        description(ui, "Keep hands, hair and glasses that cover the face. Runs one extra model per face.");
    });
}

/// Restrict the swap to chosen parts of the face.
pub(super) fn regions(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    toggle_section(ui, icons::FACE_RETOUCHING_NATURAL, "Face regions", &mut settings.region_mask, |ui| {
        description(ui, "Only replace the selected parts of the face.");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = eframe::egui::vec2(spacing::SM, spacing::SM);
            for region in FaceRegion::ALL {
                let on = settings.face_regions.contains(region);
                if ui.add(Chip::filter(region.label(), on)).clicked() {
                    settings.face_regions = settings.face_regions.with(region, !on);
                }
            }
        });
    });
}

/// Keep the original mouth for natural lip-sync.
pub(super) fn mouth(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    toggle_section(ui, icons::MOOD, "Keep mouth", &mut settings.mouth_mask, |ui| {
        description(ui, "Keep your own mouth and lips so speech and expressions stay natural.");
    });
}

/// Match the swapped face's colours to the original.
pub(super) fn color(ui: &mut Ui, frame: &mut Frame<'_>) {
    let settings = &mut *frame.settings;
    section(ui, icons::PALETTE, "Colour match", |ui| {
        description(ui, "Blend the new face's colours towards the original lighting and skin tone.");
        labeled_slider(ui, "Strength", &mut settings.color_correction, 0.0..=1.0, percent);
    });
}
