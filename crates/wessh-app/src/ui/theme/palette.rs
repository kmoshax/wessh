//! Tonal palettes: one hue/chroma pair rendered at any tone.

use super::hct::{Hct, Rgb};
use eframe::egui::Color32;

/// All tones of a single hue and chroma (MD3 "tonal palette").
#[derive(Clone, Copy, Debug)]
pub struct TonalPalette {
    hue: f64,
    chroma: f64,
}

impl TonalPalette {
    /// A palette with the given CAM16 hue (degrees) and chroma.
    #[must_use]
    pub const fn new(hue: f64, chroma: f64) -> Self {
        Self { hue, chroma }
    }

    /// The colour at `tone` (L*, 0 = black, 100 = white).
    #[must_use]
    pub fn tone(&self, tone: f64) -> Color32 {
        let [r, g, b]: Rgb = Hct::solve(self.hue, self.chroma, tone);
        Color32::from_rgb(r, g, b)
    }
}

/// The six key palettes a colour scheme is built from.
#[derive(Clone, Copy, Debug)]
pub struct CorePalettes {
    pub primary: TonalPalette,
    pub secondary: TonalPalette,
    pub tertiary: TonalPalette,
    pub neutral: TonalPalette,
    pub neutral_variant: TonalPalette,
    pub error: TonalPalette,
}

/// Chroma targets of the MD3 default "tonal spot" scheme variant.
mod tonal_spot {
    pub const PRIMARY_CHROMA: f64 = 36.0;
    pub const SECONDARY_CHROMA: f64 = 16.0;
    pub const TERTIARY_CHROMA: f64 = 24.0;
    pub const TERTIARY_HUE_SHIFT: f64 = 60.0;
    pub const NEUTRAL_CHROMA: f64 = 6.0;
    pub const NEUTRAL_VARIANT_CHROMA: f64 = 8.0;
    pub const ERROR_HUE: f64 = 25.0;
    pub const ERROR_CHROMA: f64 = 84.0;
}

impl CorePalettes {
    /// Derive the key palettes from a seed colour (MD3 tonal-spot variant).
    #[must_use]
    pub fn from_seed(seed: Color32) -> Self {
        use tonal_spot::{
            ERROR_CHROMA, ERROR_HUE, NEUTRAL_CHROMA, NEUTRAL_VARIANT_CHROMA, PRIMARY_CHROMA, SECONDARY_CHROMA, TERTIARY_CHROMA,
            TERTIARY_HUE_SHIFT,
        };
        let hue = Hct::from_rgb([seed.r(), seed.g(), seed.b()]).hue;
        Self {
            primary: TonalPalette::new(hue, PRIMARY_CHROMA),
            secondary: TonalPalette::new(hue, SECONDARY_CHROMA),
            tertiary: TonalPalette::new(hue + TERTIARY_HUE_SHIFT, TERTIARY_CHROMA),
            neutral: TonalPalette::new(hue, NEUTRAL_CHROMA),
            neutral_variant: TonalPalette::new(hue, NEUTRAL_VARIANT_CHROMA),
            error: TonalPalette::new(ERROR_HUE, ERROR_CHROMA),
        }
    }
}
