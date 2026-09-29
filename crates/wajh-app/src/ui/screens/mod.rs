//! Top-level destinations shown on the navigation rail.

pub mod live;
mod offline;

pub use live::LiveScreen;
pub use offline::offline_screen;

use crate::ui::components::Destination;
use crate::ui::icons;

/// A navigation destination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    #[default]
    Live,
    Image,
    Video,
}

impl Screen {
    /// Rail entries in order.
    pub const DESTINATIONS: [Destination<'static, Self>; 3] = [
        Destination { value: Self::Live, icon: icons::VIDEOCAM, label: "Live" },
        Destination { value: Self::Image, icon: icons::IMAGE, label: "Image" },
        Destination { value: Self::Video, icon: icons::MOVIE, label: "Video" },
    ];
}
