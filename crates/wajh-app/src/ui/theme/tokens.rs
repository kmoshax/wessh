//! MD3 design tokens: shape, spacing, elevation, motion and state layers.

use eframe::egui::{Color32, CornerRadius, Shadow};

/// MD3 corner-radius scale.
pub mod shape {
    use super::CornerRadius;

    pub const NONE: CornerRadius = CornerRadius::ZERO;
    pub const EXTRA_SMALL: CornerRadius = CornerRadius::same(4);
    pub const SMALL: CornerRadius = CornerRadius::same(8);
    pub const MEDIUM: CornerRadius = CornerRadius::same(12);
    pub const LARGE: CornerRadius = CornerRadius::same(16);
    pub const EXTRA_LARGE: CornerRadius = CornerRadius::same(28);

    /// Fully rounded ends for a component of the given height (pills, circles).
    #[must_use]
    pub fn full(height: f32) -> CornerRadius {
        CornerRadius::same((height / 2.0).round().min(f32::from(u8::MAX)) as u8)
    }
}

/// Spacing on the MD3 4dp grid.
pub mod spacing {
    pub const XXS: f32 = 2.0;
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const XL: f32 = 24.0;
    pub const XXL: f32 = 32.0;
}

/// Standard component sizes.
pub mod size {
    /// Minimum touch / pointer target.
    pub const TARGET: f32 = 48.0;
    pub const ICON: f32 = 24.0;
    pub const ICON_SMALL: f32 = 18.0;
    pub const ICON_LARGE: f32 = 40.0;
    pub const STROKE: f32 = 1.0;
    pub const FOCUS_STROKE: f32 = 2.0;
}

/// Elevation levels 0–5, rendered as a single soft key shadow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elevation {
    Level0,
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

impl Elevation {
    /// The drop shadow for this level, tinted with the scheme's `shadow` role.
    #[must_use]
    pub fn shadow(self, shadow: Color32) -> Shadow {
        const KEY_ALPHA: f32 = 0.30;
        let (offset, blur) = match self {
            Self::Level0 => return Shadow::NONE,
            Self::Level1 => (1, 3),
            Self::Level2 => (2, 6),
            Self::Level3 => (4, 8),
            Self::Level4 => (6, 10),
            Self::Level5 => (8, 12),
        };
        Shadow { offset: [0, offset], blur, spread: 0, color: shadow.gamma_multiply(KEY_ALPHA) }
    }
}

/// Motion durations (seconds) and easing.
pub mod motion {
    pub const SHORT_2: f32 = 0.100;
    pub const SHORT_4: f32 = 0.200;
    pub const MEDIUM_2: f32 = 0.300;
    pub const MEDIUM_4: f32 = 0.400;
    pub const LONG_2: f32 = 0.500;

    /// Approximation of MD3 "emphasized decelerate" easing.
    #[must_use]
    pub fn emphasized(t: f32) -> f32 {
        eframe::emath::easing::cubic_out(t)
    }

    /// Approximation of MD3 "standard" easing.
    #[must_use]
    pub fn standard(t: f32) -> f32 {
        eframe::emath::easing::cubic_in_out(t)
    }
}

/// State-layer and disabled opacities.
pub mod state {
    use super::Color32;

    pub const HOVER: f32 = 0.08;
    pub const FOCUS: f32 = 0.10;
    pub const PRESSED: f32 = 0.10;
    pub const DRAGGED: f32 = 0.16;
    pub const DISABLED_CONTENT: f32 = 0.38;
    pub const DISABLED_CONTAINER: f32 = 0.12;

    /// `base` with a translucent `layer` of `opacity` painted over it.
    #[must_use]
    pub fn overlay(base: Color32, layer: Color32, opacity: f32) -> Color32 {
        base.blend(layer.gamma_multiply(opacity))
    }

    /// A colour at the given opacity (for disabled content and containers).
    #[must_use]
    pub fn with_opacity(color: Color32, opacity: f32) -> Color32 {
        color.gamma_multiply(opacity)
    }
}
