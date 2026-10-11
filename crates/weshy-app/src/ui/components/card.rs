//! MD3 cards (filled, elevated, outlined).

use crate::ui::theme::{self, Elevation, shape, size, spacing};
use eframe::egui::{Frame, InnerResponse, Margin, Stroke, Ui};

/// Card style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardVariant {
    Filled,
    Elevated,
    Outlined,
}

/// A rounded container grouping related content.
#[must_use = "call `.show(ui, ...)`"]
pub struct Card {
    variant: CardVariant,
    padding: f32,
}

impl Card {
    /// A card of the given style with the default 16 dp padding.
    pub fn new(variant: CardVariant) -> Self {
        Self { variant, padding: spacing::LG }
    }

    /// Filled card on `surfaceContainerHighest`.
    pub fn filled() -> Self {
        Self::new(CardVariant::Filled)
    }

    /// Elevated card on `surfaceContainerLow` with a level-1 shadow.
    pub fn elevated() -> Self {
        Self::new(CardVariant::Elevated)
    }

    /// Outlined card on `surface`.
    pub fn outlined() -> Self {
        Self::new(CardVariant::Outlined)
    }

    /// Inner padding.
    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// The egui frame for this card.
    pub fn frame(&self, ui: &Ui) -> Frame {
        let colors = theme::colors(ui.ctx());
        let frame = Frame::new().corner_radius(shape::MEDIUM).inner_margin(Margin::same(self.padding as i8));
        match self.variant {
            CardVariant::Filled => frame.fill(colors.surface_container_highest),
            CardVariant::Elevated => frame.fill(colors.surface_container_low).shadow(Elevation::Level1.shadow(colors.shadow)),
            CardVariant::Outlined => frame.fill(colors.surface).stroke(Stroke::new(size::STROKE, colors.outline_variant)),
        }
    }

    /// Show `add_contents` inside the card, stretched to the available width.
    pub fn show<R>(self, ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
        self.frame(ui).show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        })
    }
}
