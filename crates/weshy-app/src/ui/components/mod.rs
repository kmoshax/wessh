//! Reusable Material Design 3 components drawn with the theme in [`super::theme`].
//!
//! The set is a small design-system library, so it offers variants (outlined and text
//! buttons, assist and filter chips, …) that current screens do not all use yet.
#![allow(dead_code, reason = "component library: not every variant is used by the current screens")]

pub mod button;
pub mod card;
pub mod chip;
pub mod compare;
pub mod fab;
pub mod feedback;
pub mod field;
pub mod list_item;
pub mod navigation;
pub mod paint;
pub mod segmented;
pub mod slider;
pub mod switch;

pub use button::{Button, IconButton};
pub use card::Card;
pub use chip::Chip;
pub use fab::{ExtendedFab, FabColor};
pub use feedback::{CircularProgress, LinearProgress, Snackbar, drop_overlay, empty_state};
pub use field::{Dropdown, TextField};
pub use list_item::ListItem;
pub use navigation::{Destination, navigation_rail, top_app_bar};
pub use segmented::SegmentedButton;
pub use slider::Slider;
pub use switch::Switch;
