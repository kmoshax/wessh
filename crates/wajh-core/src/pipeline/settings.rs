use crate::assets::SwapModel;
use crate::image::Padding;

/// User-tunable pipeline options. Cheap to clone; read once per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub swap_model: SwapModel,
    /// Swap every detected face instead of only the largest.
    pub many_faces: bool,
    /// Restore detail with GPEN after swapping.
    pub enhance: bool,
    /// How much of the restored face is blended in, `0..=1`.
    pub enhance_blend: f32,
    /// Landmark stabilisation (live video only).
    pub stabilize: bool,
    /// Feather width of the paste mask, `0..=1` of the crop.
    pub mask_blur: f32,
    pub mask_padding: Padding,
    /// Flip the input horizontally (selfie view).
    pub mirror: bool,
    /// Detector input resolution; 320 is plenty for a close-up webcam.
    pub detector_size: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            swap_model: SwapModel::default(),
            many_faces: false,
            enhance: false,
            enhance_blend: 0.8,
            stabilize: true,
            mask_blur: 0.3,
            mask_padding: [0.0; 4],
            mirror: false,
            detector_size: 640,
        }
    }
}
