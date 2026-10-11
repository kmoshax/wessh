//! CPU image kernels. They only ever touch an aligned face crop or the face's bounding box —
//! never whole-frame copies — and parallelise rows with rayon.

mod blend;
mod color;
mod detail;
mod frame;
mod mask;
mod resize;
mod sample;
mod tensor;
mod warp;

pub use blend::paste_back;
pub use color::{ColorStats, bgr_to_lab, lab_to_bgr, match_color};
pub use detail::detail;
pub use frame::Frame;
pub use mask::{Padding, box_mask, face_mask, gaussian_blur, gaussian_blur_with};
pub use resize::{resize_crop, resize_crop_into, resize_plane, resize_plane_into};
pub use tensor::{Normalization, from_nchw, letterbox_nchw, letterbox_nchw_into, letterbox_region_nchw_into, to_nchw, to_nchw_into};
pub use warp::{warp_crop, warp_crop_into, warp_plane_into};
