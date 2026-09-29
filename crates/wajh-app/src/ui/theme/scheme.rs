//! MD3 colour roles, generated from a seed colour.

use super::palette::{CorePalettes, TonalPalette};
use eframe::egui::Color32;

/// Light or dark colour scheme.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Brightness {
    #[default]
    Dark,
    Light,
}

impl Brightness {
    /// The other brightness.
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }
}

/// Every MD3 colour role. Components read colours only from here.
#[derive(Clone, Copy, Debug)]
pub struct ColorScheme {
    pub brightness: Brightness,
    pub primary: Color32,
    pub on_primary: Color32,
    pub primary_container: Color32,
    pub on_primary_container: Color32,
    pub inverse_primary: Color32,
    pub secondary: Color32,
    pub on_secondary: Color32,
    pub secondary_container: Color32,
    pub on_secondary_container: Color32,
    pub tertiary: Color32,
    pub on_tertiary: Color32,
    pub tertiary_container: Color32,
    pub on_tertiary_container: Color32,
    pub error: Color32,
    pub on_error: Color32,
    pub error_container: Color32,
    pub on_error_container: Color32,
    pub surface: Color32,
    pub surface_dim: Color32,
    pub surface_bright: Color32,
    pub surface_container_lowest: Color32,
    pub surface_container_low: Color32,
    pub surface_container: Color32,
    pub surface_container_high: Color32,
    pub surface_container_highest: Color32,
    pub on_surface: Color32,
    pub on_surface_variant: Color32,
    pub outline: Color32,
    pub outline_variant: Color32,
    pub inverse_surface: Color32,
    pub inverse_on_surface: Color32,
    pub scrim: Color32,
    pub shadow: Color32,
}

/// Tones for an accent palette: role, on-role, container, on-container.
#[derive(Clone, Copy)]
struct AccentTones {
    role: f64,
    on_role: f64,
    container: f64,
    on_container: f64,
}

/// Tone assignments for one brightness (MD3 standard contrast).
struct ToneMap {
    accent: AccentTones,
    inverse_primary: f64,
    surface: f64,
    surface_dim: f64,
    surface_bright: f64,
    container_lowest: f64,
    container_low: f64,
    container: f64,
    container_high: f64,
    container_highest: f64,
    on_surface: f64,
    on_surface_variant: f64,
    outline: f64,
    outline_variant: f64,
    inverse_surface: f64,
    inverse_on_surface: f64,
}

const DARK_TONES: ToneMap = ToneMap {
    accent: AccentTones { role: 80.0, on_role: 20.0, container: 30.0, on_container: 90.0 },
    inverse_primary: 40.0,
    surface: 6.0,
    surface_dim: 6.0,
    surface_bright: 24.0,
    container_lowest: 4.0,
    container_low: 10.0,
    container: 12.0,
    container_high: 17.0,
    container_highest: 22.0,
    on_surface: 90.0,
    on_surface_variant: 80.0,
    outline: 60.0,
    outline_variant: 30.0,
    inverse_surface: 90.0,
    inverse_on_surface: 20.0,
};

const LIGHT_TONES: ToneMap = ToneMap {
    accent: AccentTones { role: 40.0, on_role: 100.0, container: 90.0, on_container: 10.0 },
    inverse_primary: 80.0,
    surface: 98.0,
    surface_dim: 87.0,
    surface_bright: 98.0,
    container_lowest: 100.0,
    container_low: 96.0,
    container: 94.0,
    container_high: 92.0,
    container_highest: 90.0,
    on_surface: 10.0,
    on_surface_variant: 30.0,
    outline: 50.0,
    outline_variant: 80.0,
    inverse_surface: 20.0,
    inverse_on_surface: 95.0,
};

/// Black in every scheme: the base of scrims and shadows.
const ABSOLUTE_BLACK_TONE: f64 = 0.0;

impl ColorScheme {
    /// Generate the full role set for `seed` at `brightness`.
    #[must_use]
    pub fn from_seed(seed: Color32, brightness: Brightness) -> Self {
        let palettes = CorePalettes::from_seed(seed);
        let tones = match brightness {
            Brightness::Dark => &DARK_TONES,
            Brightness::Light => &LIGHT_TONES,
        };
        let accent = |palette: &TonalPalette| {
            let t = tones.accent;
            [palette.tone(t.role), palette.tone(t.on_role), palette.tone(t.container), palette.tone(t.on_container)]
        };
        let [primary, on_primary, primary_container, on_primary_container] = accent(&palettes.primary);
        let [secondary, on_secondary, secondary_container, on_secondary_container] = accent(&palettes.secondary);
        let [tertiary, on_tertiary, tertiary_container, on_tertiary_container] = accent(&palettes.tertiary);
        let [error, on_error, error_container, on_error_container] = accent(&palettes.error);
        let n = |tone| palettes.neutral.tone(tone);
        let nv = |tone| palettes.neutral_variant.tone(tone);
        Self {
            brightness,
            primary,
            on_primary,
            primary_container,
            on_primary_container,
            inverse_primary: palettes.primary.tone(tones.inverse_primary),
            secondary,
            on_secondary,
            secondary_container,
            on_secondary_container,
            tertiary,
            on_tertiary,
            tertiary_container,
            on_tertiary_container,
            error,
            on_error,
            error_container,
            on_error_container,
            surface: n(tones.surface),
            surface_dim: n(tones.surface_dim),
            surface_bright: n(tones.surface_bright),
            surface_container_lowest: n(tones.container_lowest),
            surface_container_low: n(tones.container_low),
            surface_container: n(tones.container),
            surface_container_high: n(tones.container_high),
            surface_container_highest: n(tones.container_highest),
            on_surface: n(tones.on_surface),
            on_surface_variant: nv(tones.on_surface_variant),
            outline: nv(tones.outline),
            outline_variant: nv(tones.outline_variant),
            inverse_surface: n(tones.inverse_surface),
            inverse_on_surface: n(tones.inverse_on_surface),
            scrim: n(ABSOLUTE_BLACK_TONE),
            shadow: n(ABSOLUTE_BLACK_TONE),
        }
    }

    /// `true` for the dark scheme.
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.brightness == Brightness::Dark
    }
}
