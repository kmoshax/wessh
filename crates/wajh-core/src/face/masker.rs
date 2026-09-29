//! Model-based paste masks (`FaceFusion`'s occlusion and region masks, plus a mouth keep-out).
//!
//! Every mask lives in the Arcface128 template's normalised square, so a mask computed once
//! per face applies to any crop of that template — the swap crop at any pixel-boost size and
//! the enhancer's crop alike. Masks are composed at [`MASK_SIZE`]: they are feathered anyway,
//! and at that size blurring and combining them costs well under a millisecond.
//!
//! The parser runs at most once per face per frame: the region mask and the mouth keep-out are
//! both read from the same label map.
//!
//! In live mode the masks are also reused across frames. Because they are expressed in the
//! aligned face's frame, they stay valid while the face merely moves, turns a little or scales;
//! they go stale when its content changes (expression, a hand coming in). A mask is recomputed
//! when it is [`REFRESH_INTERVAL`] frames old, when the aligned landmarks drift by
//! [`LANDMARK_DRIFT`], or when the aligned crop's content changes by [`CROP_CHANGE`]. The
//! periodic refresh of the occluder and of the parser — and of different faces — are staggered
//! so they don't land on the same frame, and a periodic refresh is cross-faded over one frame
//! so the mask doesn't visibly pop.

use super::{AlignedCrop, Face, FaceRegions, Occluder, Parser, parser::select_regions};
use crate::Result;
use crate::assets::{self, Asset, OccluderModel};
use crate::geometry::{Landmarks5, Point, Template};
use crate::image::{Frame, gaussian_blur_with, resize_crop_into, resize_plane_into, warp_crop_into};
use crate::runtime::Engine;
use std::time::Instant;

/// Resolution the model masks are composed at.
pub const MASK_SIZE: usize = 128;
/// Feather radius (Gaussian sigma) as a fraction of the crop — `FaceFusion`'s 5 px at 256.
const FEATHER: f32 = 5.0 / 256.0;
/// The mouth keep-out is blurred wider than the other masks so it covers the lips' edges.
const MOUTH_FEATHER_SCALE: f32 = 1.5;

/// Parser network and the crop size it runs at. On an RTX 4050 (CUDA), with the label-map
/// patch: `bisenet_resnet_34` at 512 px 11.4 ms, `bisenet_resnet_18` at 512 px 8.7 ms, at 384 px
/// 5.8 ms, at 256 px 3.6 ms. Against the 34-layer model at 512 px the 18-layer one at 384 px keeps
/// face-region `IoU` at 0.97–0.99 and mouth `IoU` at 0.83–0.94 (masks are composed at 128 px anyway);
/// at 256 px both models break down on some faces (holes in the skin, lost mouths).
pub const PARSER_ASSET: Asset = assets::PARSER_LITE;
pub const PARSER_SIZE: usize = 384;

/// Live mode: frames a cached model mask is reused for before it is recomputed anyway.
pub const REFRESH_INTERVAL: u32 = 6;
/// Live mode: largest landmark movement in the aligned crop (fraction of its side) a cached mask
/// tolerates — expression and head-turn changes the similarity alignment doesn't absorb. The
/// masks are feathered by ~2.5 px at 128 px, so 1.5 % (≈ 2 px) stays inside the feather.
pub const LANDMARK_DRIFT: f32 = 0.015;
/// Live mode: share of the face's thumbnail cells whose content changed (see [`CELL_CHANGE`])
/// that invalidates a cached mask — something moved in front of the face or the mouth opened.
pub const CROP_CHANGE: f32 = 0.04;
/// Luma change (0–255) that marks a thumbnail cell as changed. Below it are resampling noise
/// and lighting flicker.
const CELL_CHANGE: f32 = 30.0;
/// Side of the luma thumbnail compared for [`CROP_CHANGE`].
const THUMB: usize = 16;
/// Part of the aligned crop the thumbnail covers (fractions of its side): the face itself, from
/// the brows to the chin, around the Arcface128 landmarks. The background around the face moves
/// relative to it whenever the head moves, so it is left out.
const FACE_WINDOW_X: [f32; 2] = [0.2, 0.8];
const FACE_WINDOW_Y: [f32; 2] = [0.25, 0.9];
/// A face whose centre jumped by more than this many face sizes since the last frame is
/// treated as a different face (its cached masks are dropped).
const SAME_FACE_JUMP: f32 = 0.5;

/// Which model masks to apply. Everything off means the plain box mask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaskSpec {
    /// Keep whatever covers the face (hands, hair, glasses, microphones…).
    pub occlusion: Option<OccluderModel>,
    /// Only swap these parts of the face.
    pub regions: Option<FaceRegions>,
    /// Keep the target's own mouth and lips (lip-sync stays natural when talking).
    pub keep_mouth: bool,
}

impl MaskSpec {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.occlusion.is_none() && self.regions.is_none() && !self.keep_mouth
    }

    const fn needs_parser(&self) -> bool {
        self.regions.is_some() || self.keep_mouth
    }
}

/// Model-mask work of one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaskStats {
    /// Occluder time: crop, inference, feathering.
    pub occlusion_ms: f32,
    /// Parser time: crop, inference, region selection, feathering.
    pub parser_ms: f32,
    /// Model masks computed.
    pub computed: usize,
    /// Model masks reused from an earlier frame.
    pub reused: usize,
}

/// What a face's aligned crop looked like when a mask was computed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Reference {
    /// Landmarks in the aligned crop, as fractions of its side.
    landmarks: Landmarks5,
    /// Zero-mean luma thumbnail of the aligned crop.
    thumb: [f32; THUMB * THUMB],
}

impl Reference {
    fn new(landmarks: &Landmarks5, crop: &[u8], size: usize) -> Self {
        let to_unit = Template::Arcface128.align(landmarks, 1);
        Self { landmarks: landmarks.map(|p| to_unit.apply(p)), thumb: thumbnail(crop, size) }
    }

    /// Largest landmark movement and share of changed thumbnail cells since `self`.
    fn change(&self, now: &Self) -> (f32, f32) {
        let drift = self.landmarks.iter().zip(&now.landmarks).map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1])).fold(0.0, f32::max);
        (drift, changed_cells(&self.thumb, &now.thumb) as f32 / (THUMB * THUMB) as f32)
    }
}

/// Cells of `now` that differ by more than [`CELL_CHANGE`] from the matching cell of
/// `reference` and from all its neighbours. Sub-cell misalignment (landmark jitter, smoothing lag)
/// only moves edges into a neighbouring cell and is ignored; new content — an opening mouth, a
/// hand — matches no neighbour.
#[doc(hidden)]
#[must_use]
pub fn changed_cells(reference: &[f32; THUMB * THUMB], now: &[f32; THUMB * THUMB]) -> usize {
    let mut changed = 0;
    for y in 0..THUMB {
        for x in 0..THUMB {
            let value = now[y * THUMB + x];
            let near = |ny: usize, nx: usize| (value - reference[ny * THUMB + nx]).abs() <= CELL_CHANGE;
            let matched = (y.saturating_sub(1)..=(y + 1).min(THUMB - 1))
                .any(|ny| (x.saturating_sub(1)..=(x + 1).min(THUMB - 1)).any(|nx| near(ny, nx)));
            changed += usize::from(!matched);
        }
    }
    changed
}

/// Zero-mean luma of the face window of a `size`² BGR crop, block-averaged to `THUMB`².
#[doc(hidden)]
#[must_use]
pub fn thumbnail(crop: &[u8], size: usize) -> [f32; THUMB * THUMB] {
    let mut thumb = [0f32; THUMB * THUMB];
    let mut counts = [0u32; THUMB * THUMB];
    let window = |[start, end]: [f32; 2]| ((start * size as f32) as usize, (((end - start) * size as f32) as usize).max(1));
    let ((x0, width), (y0, height)) = (window(FACE_WINDOW_X), window(FACE_WINDOW_Y));
    for (y, row) in crop.chunks_exact(size * 3).skip(y0).take(height).enumerate() {
        let ty = y * THUMB / height;
        for (x, px) in row.as_chunks::<3>().0[x0..x0 + width].iter().enumerate() {
            let t = ty * THUMB + x * THUMB / width;
            // ITU-R BT.601 luma weights; the crop is BGR.
            thumb[t] += 0.114f32.mul_add(f32::from(px[0]), 0.587f32.mul_add(f32::from(px[1]), 0.299 * f32::from(px[2])));
            counts[t] += 1;
        }
    }
    for (t, n) in thumb.iter_mut().zip(counts) {
        *t /= n.max(1) as f32;
    }
    let mean = thumb.iter().sum::<f32>() / thumb.len() as f32;
    for t in &mut thumb {
        *t -= mean;
    }
    thumb
}

/// Why a cached mask is (not) recomputed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refresh {
    /// Still good.
    Reuse,
    /// Old enough to be refreshed; cross-fade into the new one.
    Scheduled,
    /// Missing or stale: use the new mask straight away.
    Now,
}

/// One model mask of one face, possibly from an earlier frame.
struct Cached {
    mask: Vec<f32>,
    /// Output of a cross-fading frame: the average of the previous and the new mask.
    faded: Vec<f32>,
    fading: bool,
    reference: Option<Reference>,
    /// Frames until the periodic refresh.
    due_in: u32,
}

impl Cached {
    fn new() -> Self {
        Self { mask: vec![0.0; MASK_SIZE * MASK_SIZE], faded: vec![0.0; MASK_SIZE * MASK_SIZE], fading: false, reference: None, due_in: 0 }
    }

    fn refresh(&self, now: &Reference, live: bool) -> Refresh {
        let Some(reference) = &self.reference else { return Refresh::Now };
        if !live {
            return Refresh::Now;
        }
        let (drift, diff) = reference.change(now);
        if drift > LANDMARK_DRIFT || diff > CROP_CHANGE {
            Refresh::Now
        } else if self.due_in == 0 {
            Refresh::Scheduled
        } else {
            Refresh::Reuse
        }
    }

    /// Record a newly computed `self.mask` (the previous one is in `previous`).
    fn computed(&mut self, previous: &[f32], refresh: Refresh, reference: &Reference, stagger: u32) {
        self.fading = refresh == Refresh::Scheduled;
        if self.fading {
            for ((f, &old), &new) in self.faded.iter_mut().zip(previous).zip(&self.mask) {
                *f = f32::midpoint(old, new);
            }
        }
        // The first mask of a face is phase-shifted so the periodic refreshes of different
        // masks and faces fall on different frames.
        self.due_in = if self.reference.is_none() { REFRESH_INTERVAL - stagger % REFRESH_INTERVAL } else { REFRESH_INTERVAL };
        self.reference = Some(*reference);
    }

    fn reused(&mut self) {
        self.fading = false;
        self.due_in = self.due_in.saturating_sub(1);
    }

    fn output(&self) -> &[f32] {
        if self.fading { &self.faded } else { &self.mask }
    }
}

/// Model masks of one face (by position in the frame's face list).
struct Slot {
    center: Point,
    size: f32,
    occlusion: Cached,
    /// Region mask ∩ mouth keep-out: both come from the parser.
    parsed: Cached,
    combined: Vec<f32>,
}

impl Slot {
    fn new(face: &Face) -> Self {
        Self {
            center: center(face),
            size: face.size(),
            occlusion: Cached::new(),
            parsed: Cached::new(),
            combined: vec![0.0; MASK_SIZE * MASK_SIZE],
        }
    }

    fn forget(&mut self) {
        self.occlusion.reference = None;
        self.parsed.reference = None;
    }
}

fn center(face: &Face) -> Point {
    [f32::midpoint(face.bbox[0], face.bbox[2]), f32::midpoint(face.bbox[1], face.bbox[3])]
}

/// Buffers reused for every mask.
struct Scratch {
    occluder_crop: Vec<u8>,
    parser_crop: Vec<u8>,
    plane: Vec<f32>,
    previous: Vec<f32>,
    mouth: Vec<f32>,
    kernel: Vec<f32>,
    blur: Vec<f32>,
}

impl Scratch {
    fn new(parser_size: usize) -> Self {
        Self {
            occluder_crop: vec![0; Occluder::SIZE * Occluder::SIZE * Frame::CHANNELS],
            parser_crop: vec![0; parser_size * parser_size * Frame::CHANNELS],
            plane: vec![0.0; parser_size.max(Occluder::SIZE).pow(2)],
            previous: vec![0.0; MASK_SIZE * MASK_SIZE],
            mouth: vec![0.0; MASK_SIZE * MASK_SIZE],
            kernel: Vec::new(),
            blur: Vec::new(),
        }
    }
}

/// Lazily loaded occluder and parser models, plus each face's cached masks.
pub struct Masker {
    occluder: Option<Occluder>,
    parser: Option<Parser>,
    /// Reuse masks across live frames; `WAJH_MASK_CACHE=0` turns it off (for measurements).
    cache: bool,
    slots: Vec<Slot>,
    /// Spec the cached masks were made with.
    spec: MaskSpec,
    scratch: Scratch,
    stats: MaskStats,
}

impl Default for Masker {
    fn default() -> Self {
        Self {
            occluder: None,
            parser: None,
            cache: std::env::var_os("WAJH_MASK_CACHE").is_none_or(|v| v != "0"),
            slots: Vec::new(),
            spec: MaskSpec::default(),
            scratch: Scratch::new(PARSER_SIZE),
            stats: MaskStats::default(),
        }
    }
}

impl Masker {
    /// Load the models `spec` needs (downloading them on first use).
    pub fn prepare(&mut self, engine: &Engine, spec: &MaskSpec) -> Result<()> {
        if let Some(kind) = spec.occlusion
            && self.occluder.as_ref().map(Occluder::kind) != Some(kind)
        {
            self.occluder = None;
            self.occluder = Some(Occluder::new(engine, kind)?);
        }
        if spec.needs_parser() && self.parser.is_none() {
            self.parser = Some(Parser::new(engine, &PARSER_ASSET, PARSER_SIZE)?);
        }
        Ok(())
    }

    /// Start a frame: reset the per-frame stats and drop cached masks made for another spec.
    pub fn begin_frame(&mut self, spec: &MaskSpec) {
        self.stats = MaskStats::default();
        if *spec != self.spec {
            self.spec = *spec;
            self.slots.iter_mut().for_each(Slot::forget);
        }
    }

    /// Model-mask work since [`Masker::begin_frame`].
    #[must_use]
    pub const fn stats(&self) -> MaskStats {
        self.stats
    }

    /// Update the model mask of face number `index` of the frame and return it —
    /// `MASK_SIZE`² in the Arcface128 template — or `None` when `spec` is empty.
    ///
    /// `original` is the aligned target crop the swap replaced (`swapped.size`²) and `swapped`
    /// the swap output. With `live`, masks from earlier frames are reused while they are still
    /// valid; otherwise every mask is computed fresh.
    pub fn face_mask(
        &mut self,
        index: usize,
        frame: &Frame,
        face: &Face,
        original: &[u8],
        swapped: &AlignedCrop,
        live: bool,
    ) -> Result<Option<&[f32]>> {
        let spec = self.spec;
        let live = live && self.cache;
        if spec.is_empty() {
            return Ok(None);
        }
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || Slot::new(face));
        }
        let slot = &mut self.slots[index];
        let (moved, size) = (center(face), face.size());
        if (moved[0] - slot.center[0]).hypot(moved[1] - slot.center[1]) > SAME_FACE_JUMP * slot.size.max(size) {
            slot.forget();
        }
        (slot.center, slot.size) = (moved, size);
        let now = Reference::new(&face.landmarks, original, swapped.size);
        let scratch = &mut self.scratch;
        let sigma = FEATHER * MASK_SIZE as f32;
        // Stagger: faces by one frame each, the parser half an interval after the occluder.
        let stagger = index as u32;

        let mut have_mask = false;
        if spec.occlusion.is_some()
            && let Some(occluder) = &mut self.occluder
        {
            let cached = &mut slot.occlusion;
            let refresh = cached.refresh(&now, live);
            if refresh == Refresh::Reuse {
                cached.reused();
                self.stats.reused += 1;
            } else {
                let started = Instant::now();
                let crop = &mut scratch.occluder_crop;
                warp_crop_into(frame, &Template::Arcface128.align(&face.landmarks, Occluder::SIZE), Occluder::SIZE, crop);
                scratch.previous.copy_from_slice(&cached.mask);
                resize_plane_into(occluder.visible(crop)?, Occluder::SIZE, MASK_SIZE, &mut cached.mask);
                feather(&mut cached.mask, MASK_SIZE, sigma, &mut scratch.kernel, &mut scratch.blur);
                cached.computed(&scratch.previous, refresh, &now, stagger);
                self.stats.computed += 1;
                self.stats.occlusion_ms += elapsed_ms(started);
            }
            slot.combined.copy_from_slice(cached.output());
            have_mask = true;
        }
        if spec.needs_parser()
            && let Some(parser) = &mut self.parser
        {
            let cached = &mut slot.parsed;
            let refresh = cached.refresh(&now, live);
            if refresh == Refresh::Reuse {
                cached.reused();
                self.stats.reused += 1;
            } else {
                let started = Instant::now();
                let size = parser.size();
                let crop = &mut scratch.parser_crop;
                if spec.regions.is_some() {
                    // Like FaceFusion, parse the swapped face: its features decide what gets
                    // pasted. Its mouth is where the target's is (the swap keeps expression), so
                    // the mouth keep-out comes from the same parse.
                    resize_crop_into(&swapped.pixels, swapped.size, size, crop);
                } else {
                    warp_crop_into(frame, &Template::Arcface128.align(&face.landmarks, size), size, crop);
                }
                let labels = parser.parse(crop)?;
                let plane = &mut scratch.plane[..size * size];
                scratch.previous.copy_from_slice(&cached.mask);
                let mut first = true;
                if let Some(regions) = spec.regions {
                    select_regions(labels, regions, plane);
                    resize_plane_into(plane, size, MASK_SIZE, &mut cached.mask);
                    feather(&mut cached.mask, MASK_SIZE, sigma, &mut scratch.kernel, &mut scratch.blur);
                    first = false;
                }
                if spec.keep_mouth {
                    select_regions(labels, FaceRegions::MOUTH, plane);
                    let mouth = &mut scratch.mouth;
                    resize_plane_into(plane, size, MASK_SIZE, mouth);
                    keep_out(mouth, MASK_SIZE, sigma * MOUTH_FEATHER_SCALE, &mut scratch.kernel, &mut scratch.blur);
                    if first {
                        cached.mask.copy_from_slice(mouth);
                    } else {
                        intersect(&mut cached.mask, mouth);
                    }
                }
                cached.computed(&scratch.previous, refresh, &now, stagger + REFRESH_INTERVAL / 2);
                self.stats.computed += 1;
                self.stats.parser_ms += elapsed_ms(started);
            }
            if have_mask {
                intersect(&mut slot.combined, cached.output());
            } else {
                slot.combined.copy_from_slice(cached.output());
            }
            have_mask = true;
        }
        Ok(have_mask.then_some(slot.combined.as_slice()))
    }

    /// The mask [`Masker::face_mask`] last returned for face `index`.
    #[must_use]
    pub fn mask(&self, index: usize) -> Option<&[f32]> {
        (!self.spec.is_empty()).then(|| self.slots.get(index).map(|s| s.combined.as_slice())).flatten()
    }
}

fn elapsed_ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}

/// Blur a hard mask and remap `[0.5, 1] → [0, 1]`: soft edges that stay inside the region
/// (`FaceFusion`'s `(blur(m).clip(0.5, 1) - 0.5) * 2`).
fn feather(mask: &mut [f32], size: usize, sigma: f32, kernel: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    clamp_unit(mask);
    gaussian_blur_with(mask, size, size, sigma, kernel, tmp);
    for v in mask {
        *v = ((v.clamp(0.5, 1.0) - 0.5) * 2.0).clamp(0.0, 1.0);
    }
}

/// Turn a hard "keep the original here" region into a paste mask: blurred, grown so the whole
/// region is kept (`[0, 0.5] → [0, 1]`), then inverted.
fn keep_out(mask: &mut [f32], size: usize, sigma: f32, kernel: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    clamp_unit(mask);
    gaussian_blur_with(mask, size, size, sigma, kernel, tmp);
    for v in mask {
        *v = 1.0 - (*v * 2.0).clamp(0.0, 1.0);
    }
}

fn clamp_unit(mask: &mut [f32]) {
    for v in mask {
        *v = v.clamp(0.0, 1.0);
    }
}

/// `acc = min(acc, other)` — masks combine by intersection.
pub fn intersect(acc: &mut [f32], other: &[f32]) {
    acc.iter_mut().zip(other).for_each(|(a, &b)| *a = a.min(b));
}

/// The final paste mask at `size` into `out`: the box mask intersected with a model mask (any
/// template resolution), if there is one. `scratch` holds the resampled model mask.
pub fn compose_into(box_mask: &[f32], size: usize, model_mask: Option<(&[f32], usize)>, out: &mut Vec<f32>, scratch: &mut Vec<f32>) {
    out.clear();
    out.extend_from_slice(box_mask);
    if let Some((model_mask, model_size)) = model_mask {
        scratch.resize(size * size, 0.0);
        resize_plane_into(model_mask, model_size, size, scratch);
        intersect(out, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(size: usize, radius: f32) -> Vec<f32> {
        let c = size as f32 / 2.0;
        (0..size * size).map(|i| f32::from(((i % size) as f32 - c).hypot((i / size) as f32 - c) < radius)).collect()
    }

    fn compose(box_mask: &[f32], size: usize, model_mask: Option<(&[f32], usize)>) -> Vec<f32> {
        let mut out = Vec::new();
        compose_into(box_mask, size, model_mask, &mut out, &mut Vec::new());
        out
    }

    #[test]
    fn composition_is_the_intersection() {
        let a = [0.0, 0.5, 1.0, 1.0];
        let b = [1.0, 1.0, 0.25, 1.0];
        assert_eq!(compose(&a, 2, Some((&b, 2))), [0.0, 0.5, 0.25, 1.0]);
        assert_eq!(compose(&a, 2, None), a);
        // A coarser model mask is resampled to the paste size first.
        let upsampled = compose(&[1.0; 16], 4, Some((&[0.0, 1.0, 0.0, 1.0], 2)));
        assert!(upsampled[0] == 0.0 && upsampled[3] == 1.0);
    }

    #[test]
    fn feather_keeps_inside_and_softens_edges() {
        let mut m = disc(64, 20.0);
        feather(&mut m, 64, 2.0, &mut Vec::new(), &mut Vec::new());
        assert!(m[32 * 64 + 32] > 0.999);
        assert_eq!(m[0], 0.0);
        // The soft edge sits inside the original region, never outside it.
        assert!(m.iter().zip(disc(64, 20.0)).all(|(f, d)| *f <= d + 1e-6));
        assert!(m.iter().any(|v| *v > 0.1 && *v < 0.9));
    }

    #[test]
    fn keep_out_covers_the_whole_region() {
        let mut m = disc(64, 10.0);
        keep_out(&mut m, 64, 2.0, &mut Vec::new(), &mut Vec::new());
        // Up to its (curved, pixelated) boundary the region is fully kept.
        assert!(m.iter().zip(disc(64, 9.0)).all(|(k, r)| r == 0.0 || *k == 0.0));
        assert_eq!(m[0], 1.0);
    }

    fn reference(shift: f32, brightness: u8) -> Reference {
        let landmarks = Template::Arcface128.points().map(|[x, y]| [x * 100.0 + shift, y * 100.0]);
        let size = 32;
        // A vertical gradient so the thumbnail has structure.
        let crop: Vec<u8> = (0..size * size).flat_map(|i| [((i / size) * 4) as u8 / 2 + brightness; 3]).collect();
        Reference::new(&landmarks, &crop, size)
    }

    #[test]
    fn rigid_motion_and_lighting_do_not_invalidate() {
        let (drift, diff) = reference(0.0, 0).change(&reference(25.0, 40));
        assert!(drift < 1e-4 && diff == 0.0, "{drift} {diff}");
    }

    #[test]
    fn cache_refreshes_on_schedule_on_change_and_outside_live_mode() {
        let still = reference(0.0, 0);
        let mut cached = Cached::new();
        assert_eq!(cached.refresh(&still, true), Refresh::Now);
        cached.computed(&[], Refresh::Now, &still, 0);
        let mut schedule = Vec::new();
        for _ in 0..=REFRESH_INTERVAL {
            let refresh = cached.refresh(&still, true);
            schedule.push(refresh);
            if refresh == Refresh::Reuse {
                cached.reused();
            } else {
                cached.computed(&vec![0.0; MASK_SIZE * MASK_SIZE], refresh, &still, 0);
            }
        }
        assert_eq!(schedule.iter().filter(|r| **r == Refresh::Reuse).count(), REFRESH_INTERVAL as usize);
        assert_eq!(schedule[REFRESH_INTERVAL as usize], Refresh::Scheduled);
        assert!(cached.fading);

        // A mouth opening changes the landmarks inside the aligned crop.
        let mut open = still;
        open.landmarks[3][1] += 2.0 * LANDMARK_DRIFT;
        assert_eq!(cached.refresh(&open, true), Refresh::Now);
        // A hand in front of the face changes the crop.
        let mut covered = still;
        covered.thumb[..THUMB * THUMB / 2].iter_mut().for_each(|t| *t += 2.0 * CELL_CHANGE);
        assert_eq!(cached.refresh(&covered, true), Refresh::Now);
        assert_eq!(cached.refresh(&still, false), Refresh::Now);
    }

    #[test]
    fn stagger_shifts_the_first_refresh() {
        let still = reference(0.0, 0);
        let mut a = Cached::new();
        let mut b = Cached::new();
        a.computed(&[], Refresh::Now, &still, 0);
        b.computed(&[], Refresh::Now, &still, REFRESH_INTERVAL / 2);
        assert_ne!(a.due_in, b.due_in);
    }

    #[test]
    fn scheduled_refresh_cross_fades() {
        let still = reference(0.0, 0);
        let mut cached = Cached::new();
        cached.computed(&[], Refresh::Now, &still, 0);
        let previous = vec![0.0; MASK_SIZE * MASK_SIZE];
        cached.mask.fill(1.0);
        cached.computed(&previous, Refresh::Scheduled, &still, 0);
        assert!(cached.output().iter().all(|v| *v == 0.5));
        cached.reused();
        assert!(cached.output().iter().all(|v| *v == 1.0));
    }
}
