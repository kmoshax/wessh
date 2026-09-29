use super::FaceOrder;
use crate::assets::{OccluderModel, SwapModel};
use crate::face::{FaceRegions, MaskSpec};
use crate::image::Padding;

/// User-tunable pipeline options. Cheap to clone; read once per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub swap_model: SwapModel,
    /// Output size of the swapped face. Above the model's native resolution the model runs on
    /// sub-sampled tiles (`FaceFusion`'s pixel boost): sharper, at `(size / native)²` swaps per face.
    /// Snapped down to a multiple of the native resolution; `0` means native. Off by default
    /// because it multiplies the swap cost — use 256–512 for images and offline video.
    pub pixel_boost: usize,
    /// Swap every detected face instead of only the largest.
    pub many_faces: bool,
    /// Live video: run the swap network on at most this many faces per frame; the others take
    /// turns and meanwhile reuse their last swapped face, re-warped to where they are now (the
    /// expression lags a few frames). `0` swaps every face every frame.
    pub max_faces_per_frame: usize,
    /// Order faces are picked in (with `many_faces` off, the first one is swapped).
    pub face_order: FaceOrder,
    /// With a reference face set, only swap faces whose identity distance (`1 - cosine
    /// similarity`) to it is below this.
    pub reference_distance: f32,
    /// Restore detail with GPEN after swapping.
    pub enhance: bool,
    /// How much of the restored face is blended in, `0..=1`.
    pub enhance_blend: f32,
    /// Landmark stabilisation (live video only).
    pub stabilize: bool,
    /// Feather width of the paste mask, `0..=1` of the crop.
    pub mask_blur: f32,
    pub mask_padding: Padding,
    /// Keep hands, hair and glasses that cover the face (`XSeg` occlusion mask). Costs one occluder
    /// inference per face, so it is off by default.
    pub occlusion: bool,
    pub occluder_model: OccluderModel,
    /// Only swap the parser regions in `face_regions` (`BiSeNet` region mask).
    pub region_mask: bool,
    pub face_regions: FaceRegions,
    /// Keep the target's own mouth and lips — natural lip-sync when talking.
    pub mouth_mask: bool,
    /// Match the swapped face's colours to the original face, `0..=1` (0 = off).
    pub color_correction: f32,
    /// Flip the input horizontally (selfie view).
    pub mirror: bool,
    /// Detector input resolution; 320 is plenty for a close-up webcam.
    pub detector_size: usize,
}

impl Settings {
    /// The model masks these settings ask for.
    #[must_use]
    pub fn mask_spec(&self) -> MaskSpec {
        MaskSpec {
            occlusion: self.occlusion.then_some(self.occluder_model),
            regions: self.region_mask.then_some(self.face_regions),
            keep_mouth: self.mouth_mask,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            swap_model: SwapModel::default(),
            pixel_boost: 0,
            many_faces: false,
            max_faces_per_frame: 0,
            face_order: FaceOrder::default(),
            reference_distance: 0.6,
            enhance: true,
            enhance_blend: 0.8,
            stabilize: true,
            mask_blur: 0.3,
            mask_padding: [0.0; 4],
            occlusion: true,
            occluder_model: OccluderModel::default(),
            region_mask: false,
            face_regions: FaceRegions::ALL,
            mouth_mask: false,
            color_correction: 0.5,
            mirror: false,
            detector_size: 640,
        }
    }
}
