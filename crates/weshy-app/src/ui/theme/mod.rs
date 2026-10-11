//! Material Design 3 theme: colour scheme, type scale, tokens and their mapping onto egui.
//!
//! The active [`ColorScheme`] is stored in egui's context memory so every component
//! reads its colours through [`colors`] instead of hard-coding them.
#![allow(dead_code, reason = "the full MD3 role and token set is kept even where screens do not use it yet")]

mod hct;
mod palette;
mod scheme;
pub mod tokens;
mod typography;
mod visuals;

pub use scheme::{Brightness, ColorScheme};
pub use tokens::{Elevation, motion, shape, size, spacing, state};
pub use typography::{TypeRole, icon_font, install_fonts};

use eframe::egui::{self, Color32, Id};

/// Seed colour the palettes are generated from (MD3 baseline violet).
pub const DEFAULT_SEED: Color32 = Color32::from_rgb(0x67, 0x50, 0xA4);

fn scheme_id() -> Id {
    Id::new("md3-color-scheme")
}

/// The seed and brightness a scheme was generated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThemeConfig {
    pub seed: Color32,
    pub brightness: Brightness,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self { seed: DEFAULT_SEED, brightness: Brightness::Dark }
    }
}

/// Generate the scheme for `config` and make it the active style.
pub fn apply(ctx: &egui::Context, config: ThemeConfig) {
    let colors = ColorScheme::from_seed(config.seed, config.brightness);
    ctx.set_style(visuals::style(&colors));
    ctx.data_mut(|data| data.insert_temp(scheme_id(), colors));
}

/// The active colour scheme (generated on demand if none was applied yet).
#[must_use]
pub fn colors(ctx: &egui::Context) -> ColorScheme {
    ctx.data(|data| data.get_temp(scheme_id())).unwrap_or_else(|| ColorScheme::from_seed(DEFAULT_SEED, Brightness::default()))
}
