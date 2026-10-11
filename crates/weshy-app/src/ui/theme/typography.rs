//! MD3 type scale on embedded Roboto, plus the Material Symbols icon fonts.

use eframe::egui::text::{LayoutJob, TextFormat};
use eframe::egui::{self, Color32, FontData, FontDefinitions, FontFamily, FontId, TextStyle};
use std::collections::BTreeMap;
use std::sync::Arc;

const ROBOTO_REGULAR: &[u8] = include_bytes!("../../../assets/fonts/Roboto-Regular.ttf");
const ROBOTO_MEDIUM: &[u8] = include_bytes!("../../../assets/fonts/Roboto-Medium.ttf");
const SYMBOLS_OUTLINED: &[u8] = include_bytes!("../../../assets/fonts/MaterialSymbolsRounded-Outlined.ttf");
const SYMBOLS_FILLED: &[u8] = include_bytes!("../../../assets/fonts/MaterialSymbolsRounded-Filled.ttf");

const REGULAR_FAMILY: &str = "roboto-regular";
const MEDIUM_FAMILY: &str = "roboto-medium";
const ICON_FAMILY: &str = "symbols-outlined";
const ICON_FILLED_FAMILY: &str = "symbols-filled";

/// Roboto weight used by a type role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Regular,
    Medium,
}

impl Weight {
    fn family(self) -> FontFamily {
        match self {
            Self::Regular => FontFamily::Proportional,
            Self::Medium => FontFamily::Name(MEDIUM_FAMILY.into()),
        }
    }
}

/// The fifteen MD3 type roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeRole {
    DisplayLarge,
    DisplayMedium,
    DisplaySmall,
    HeadlineLarge,
    HeadlineMedium,
    HeadlineSmall,
    TitleLarge,
    TitleMedium,
    TitleSmall,
    BodyLarge,
    BodyMedium,
    BodySmall,
    LabelLarge,
    LabelMedium,
    LabelSmall,
}

/// Size, line height, weight and tracking of one role.
#[derive(Clone, Copy, Debug)]
pub struct TypeStyle {
    pub size: f32,
    pub line_height: f32,
    pub weight: Weight,
    pub tracking: f32,
}

impl TypeRole {
    pub const ALL: [Self; 15] = [
        Self::DisplayLarge,
        Self::DisplayMedium,
        Self::DisplaySmall,
        Self::HeadlineLarge,
        Self::HeadlineMedium,
        Self::HeadlineSmall,
        Self::TitleLarge,
        Self::TitleMedium,
        Self::TitleSmall,
        Self::BodyLarge,
        Self::BodyMedium,
        Self::BodySmall,
        Self::LabelLarge,
        Self::LabelMedium,
        Self::LabelSmall,
    ];

    /// The MD3 baseline type scale.
    #[must_use]
    pub const fn style(self) -> TypeStyle {
        use Weight::{Medium, Regular};
        let (size, line_height, weight, tracking) = match self {
            Self::DisplayLarge => (57.0, 64.0, Regular, -0.25),
            Self::DisplayMedium => (45.0, 52.0, Regular, 0.0),
            Self::DisplaySmall => (36.0, 44.0, Regular, 0.0),
            Self::HeadlineLarge => (32.0, 40.0, Regular, 0.0),
            Self::HeadlineMedium => (28.0, 36.0, Regular, 0.0),
            Self::HeadlineSmall => (24.0, 32.0, Regular, 0.0),
            Self::TitleLarge => (22.0, 28.0, Regular, 0.0),
            Self::TitleMedium => (16.0, 24.0, Medium, 0.15),
            Self::BodyLarge => (16.0, 24.0, Regular, 0.5),
            Self::BodyMedium => (14.0, 20.0, Regular, 0.25),
            Self::BodySmall => (12.0, 16.0, Regular, 0.4),
            Self::TitleSmall | Self::LabelLarge => (14.0, 20.0, Medium, 0.1),
            Self::LabelMedium => (12.0, 16.0, Medium, 0.5),
            Self::LabelSmall => (11.0, 16.0, Medium, 0.5),
        };
        TypeStyle { size, line_height, weight, tracking }
    }

    /// Name under which the role is registered as an egui [`TextStyle`].
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DisplayLarge => "display-large",
            Self::DisplayMedium => "display-medium",
            Self::DisplaySmall => "display-small",
            Self::HeadlineLarge => "headline-large",
            Self::HeadlineMedium => "headline-medium",
            Self::HeadlineSmall => "headline-small",
            Self::TitleLarge => "title-large",
            Self::TitleMedium => "title-medium",
            Self::TitleSmall => "title-small",
            Self::BodyLarge => "body-large",
            Self::BodyMedium => "body-medium",
            Self::BodySmall => "body-small",
            Self::LabelLarge => "label-large",
            Self::LabelMedium => "label-medium",
            Self::LabelSmall => "label-small",
        }
    }

    /// The registered egui text style for this role.
    #[must_use]
    pub fn text_style(self) -> TextStyle {
        TextStyle::Name(self.name().into())
    }

    /// Font for this role.
    #[must_use]
    pub fn font_id(self) -> FontId {
        let style = self.style();
        FontId::new(style.size, style.weight.family())
    }

    /// Text format with the role's line height and tracking.
    #[must_use]
    pub fn format(self, color: Color32) -> TextFormat {
        let style = self.style();
        TextFormat {
            font_id: self.font_id(),
            extra_letter_spacing: style.tracking,
            line_height: Some(style.line_height),
            color,
            ..TextFormat::default()
        }
    }

    /// A single-run layout job, optionally wrapped at `wrap_width`.
    #[must_use]
    pub fn job(self, text: &str, color: Color32, wrap_width: Option<f32>) -> LayoutJob {
        let mut job = LayoutJob::single_section(text.to_owned(), self.format(color));
        if let Some(width) = wrap_width {
            job.wrap.max_width = width;
        }
        job
    }

    /// Lay out `text` in this role without wrapping.
    #[must_use]
    pub fn galley(self, ui: &egui::Ui, text: &str, color: Color32) -> Arc<egui::Galley> {
        ui.fonts_mut(|fonts| fonts.layout_job(self.job(text, color, None)))
    }

    /// Rich text in this role, for use with ordinary egui widgets.
    #[must_use]
    pub fn rich(self, text: impl Into<String>, color: Color32) -> egui::RichText {
        egui::RichText::new(text).text_style(self.text_style()).color(color)
    }
}

/// Font for a Material Symbol at `size`.
#[must_use]
pub fn icon_font(size: f32, filled: bool) -> FontId {
    let family = if filled { ICON_FILLED_FAMILY } else { ICON_FAMILY };
    FontId::new(size, FontFamily::Name(family.into()))
}

/// Register Roboto and Material Symbols with egui.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let mut add = |name: &str, bytes: &'static [u8]| {
        fonts.font_data.insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    };
    add(REGULAR_FAMILY, ROBOTO_REGULAR);
    add(MEDIUM_FAMILY, ROBOTO_MEDIUM);
    add(ICON_FAMILY, SYMBOLS_OUTLINED);
    add(ICON_FILLED_FAMILY, SYMBOLS_FILLED);

    let fallbacks = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let with_fallbacks = |primary: &str| {
        let mut list = vec![primary.to_owned(), ICON_FAMILY.to_owned()];
        list.extend(fallbacks.iter().cloned());
        list
    };
    fonts.families.insert(FontFamily::Proportional, with_fallbacks(REGULAR_FAMILY));
    fonts.families.insert(FontFamily::Name(MEDIUM_FAMILY.into()), with_fallbacks(MEDIUM_FAMILY));
    fonts.families.insert(FontFamily::Name(ICON_FAMILY.into()), vec![ICON_FAMILY.to_owned()]);
    fonts.families.insert(FontFamily::Name(ICON_FILLED_FAMILY.into()), vec![ICON_FILLED_FAMILY.to_owned()]);
    ctx.set_fonts(fonts);
}

/// All named MD3 text styles plus egui's built-in styles mapped onto the scale.
#[must_use]
pub fn text_styles() -> BTreeMap<TextStyle, FontId> {
    let mut styles: BTreeMap<TextStyle, FontId> = TypeRole::ALL.iter().map(|role| (role.text_style(), role.font_id())).collect();
    styles.insert(TextStyle::Small, TypeRole::BodySmall.font_id());
    styles.insert(TextStyle::Body, TypeRole::BodyMedium.font_id());
    styles.insert(TextStyle::Button, TypeRole::LabelLarge.font_id());
    styles.insert(TextStyle::Heading, TypeRole::TitleLarge.font_id());
    styles.insert(TextStyle::Monospace, FontId::monospace(TypeRole::BodyMedium.style().size));
    styles
}
