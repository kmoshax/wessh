use super::async_swap::{self, AsyncSwap, SharedSwapper};
use super::flow::LandmarkFlow;
use super::pixel_boost::{self, Swapped};
use super::reuse::SwapReuse;
use super::tracker::{FaceTracker, TrackingDetector};
use super::{FaceOrder, Settings, Stats, select};
use crate::assets::SwapModel;
use crate::face::masker::{MASK_SIZE, compose_into};
use crate::face::{AlignedCrop, Detector, Embedding, Enhancer, Face, Masker, Recognizer, Swapper};
use crate::geometry::{Affine, Rect, Template};
use crate::image::{Frame, Padding, box_mask, detail, face_mask, match_color, paste_back, warp_crop, warp_crop_into, warp_plane_into};
use crate::runtime::{Backend, Engine};
use crate::{Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A face whose every 16 px block changed by less than this (0–255, see [`SwapReuse::change`])
/// since its last swap is not swapped again: the swap would come out the same. A still face
/// measures 0 through the swap's own alignment and a repeated video frame under 2, while
/// talking, blinking and turning measure 7 and up (`demo.gif` played live).
const SAME_FACE_CHANGE: f32 = 3.0;
/// A face is swapped again at least this often however still it is, so a slow change (light
/// fading, the camera's exposure settling) can't build up unnoticed.
const MAX_SWAP_AGE: std::time::Duration = std::time::Duration::from_millis(500);

/// Async timewarp ([`super::async_swap`]) in live mode, unless `WESSH_TIMEWARP=0` (to measure
/// the synchronous pipeline).
fn timewarp_enabled() -> bool {
    std::env::var("WESSH_TIMEWARP").map_or(true, |v| v != "0")
}

/// Geometric paste masks: an ellipse for swapped faces, a feathered box for restored ones
/// (whose whole crop is the same face, just sharper).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Face,
    Box,
}

type MaskKey = (Shape, usize, u32, [u32; 4]);
/// Shape masks kept: enough for the swap and enhance crops, small enough that dragging a
/// blur/padding slider can't grow the cache without bound.
const MASK_CACHE: usize = 4;

/// Recently used shape masks, most recent first.
struct ShapeMasks(VecDeque<(MaskKey, Vec<f32>)>);

impl ShapeMasks {
    fn get(&mut self, shape: Shape, size: usize, blur: f32, padding: Padding) -> &[f32] {
        let key = (shape, size, blur.to_bits(), padding.map(f32::to_bits));
        let entry = if let Some(i) = self.0.iter().position(|(k, _)| *k == key) {
            self.0.remove(i).expect("index in range")
        } else {
            self.0.truncate(MASK_CACHE - 1);
            let mask = match shape {
                Shape::Face => face_mask(size, blur, padding),
                Shape::Box => box_mask(size, blur, padding),
            };
            (key, mask)
        };
        self.0.push_front(entry);
        &self.0[0].1
    }
}

/// Side of the square whose detail stands for a face's, as a fraction of the face size: the
/// eyes, nose and mouth, without the hair and background around them.
const DETAIL_SPAN: f32 = 0.7;
/// Live video: weight of the newest frame in the smoothed restoration amount, so it doesn't
/// shimmer with frame-to-frame noise in the detail measurements.
const AMOUNT_SMOOTHING: f32 = 0.25;
/// A swapped face with at least this share of the original's detail is left unrestored. Video
/// faces land around 0.87 (and look right without restoration; restored, they stand out), while
/// large sharp faces come out of the 128–256 px swap models at 0.2–0.7.
const DETAIL_TOLERANCE: f32 = 0.8;
/// Restoration amounts below this are skipped, along with the restorer's inference.
const MIN_AMOUNT: f32 = 0.02;

/// How much GFPGAN detail each face gets: just enough to give the swapped face as much fine
/// detail as the original face had in this frame. A small, soft or compressed face (most video)
/// then gets none — restored detail would stand out from the frame like a sticker, and the
/// restorer's inference (≈150 ms per face on a laptop GPU) is skipped — while a large, sharp
/// face gets the full restoration.
#[derive(Default)]
struct Restoration {
    /// Detail of each face before it was swapped.
    original: Vec<f32>,
    /// Live video: the smoothed amount per face.
    amounts: Vec<f32>,
    /// The face's region with the restored face pasted at full strength, to measure it.
    trial: Frame,
}

impl Restoration {
    /// Blend `needed` into face `index`'s smoothed amount (live) or take it as is.
    fn settle(&mut self, index: usize, faces: usize, needed: f32, live: bool) -> f32 {
        if !live {
            return needed;
        }
        self.amounts.resize(faces, needed);
        let amount = &mut self.amounts[index];
        *amount = AMOUNT_SMOOTHING.mul_add(needed - *amount, *amount);
        *amount
    }
}

/// The square [`detail`] is measured in for `face`.
fn detail_region(frame: &Frame, face: &Face) -> Option<Rect> {
    let [x0, y0, x1, y1] = face.bbox;
    Rect::square_around([f32::midpoint(x0, x1), f32::midpoint(y0, y1)], face.size() * DETAIL_SPAN, frame.width, frame.height)
}

/// Buffers for the final paste mask, reused every frame.
#[derive(Default)]
struct PasteMask {
    mask: Vec<f32>,
    resampled: Vec<f32>,
    /// A model mask moved from the swap's alignment into the enhancer's.
    realigned: Vec<f32>,
}

/// Owns the models and per-stream state; call [`Processor::process`] once per frame.
pub struct Processor {
    engine: Engine,
    detector: Detector,
    recognizer: Recognizer,
    swapper: SharedSwapper,
    /// The model in `swapper`, known without locking it (the swap thread holds the lock while it
    /// runs).
    swap_model: Option<SwapModel>,
    /// Live mode's swap thread, started on the first live frame.
    async_swap: Option<AsyncSwap>,
    enhancer: Option<Enhancer>,
    masker: Masker,
    identity: Option<Embedding>,
    /// `identity` converted for the current swapper.
    condition: Option<Arc<[f32]>>,
    /// Only faces like this one are swapped, when set.
    reference: Option<Embedding>,
    flow: LandmarkFlow,
    tracker: FaceTracker,
    shape_masks: ShapeMasks,
    paste_mask: PasteMask,
    reuse: SwapReuse,
    restoration: Restoration,
    /// Which faces of the frame are swapped afresh.
    fresh: Vec<bool>,
    /// A reused face's crop being pasted.
    reused_crop: Vec<u8>,
}

impl Processor {
    pub fn new(engine: Engine) -> Result<Self> {
        Ok(Self {
            detector: Detector::new(&engine)?,
            recognizer: Recognizer::new(&engine)?,
            engine,
            swapper: Arc::new(Mutex::new(None)),
            swap_model: None,
            async_swap: None,
            enhancer: None,
            masker: Masker::default(),
            identity: None,
            condition: None,
            reference: None,
            flow: LandmarkFlow::default(),
            tracker: FaceTracker::default(),
            shape_masks: ShapeMasks(VecDeque::with_capacity(MASK_CACHE)),
            paste_mask: PasteMask::default(),
            reuse: SwapReuse::default(),
            restoration: Restoration::default(),
            fresh: Vec::new(),
            reused_crop: Vec::new(),
        })
    }

    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.engine.backend()
    }

    /// Use the largest face in `image` as the identity to swap in.
    pub fn set_source(&mut self, image: &Frame) -> Result<()> {
        self.set_sources(std::slice::from_ref(image))
    }

    /// Use the average identity of the largest face in each image — several photos of the same
    /// person give a steadier, more faithful identity than one.
    pub fn set_sources(&mut self, images: &[Frame]) -> Result<()> {
        let embeddings = images.iter().map(|image| self.largest_face_identity(image)).collect::<Result<Vec<_>>>()?;
        let identity = Embedding::average(&embeddings).ok_or(Error::NoSourceFace)?;
        self.condition = async_swap::lock(&self.swapper).as_ref().map(|s| s.condition(&identity).into());
        self.identity = Some(identity);
        self.reuse.invalidate_all();
        Ok(())
    }

    #[must_use]
    pub const fn has_source(&self) -> bool {
        self.identity.is_some()
    }

    /// Only swap faces that look like face `index` (in `order`) of `image`.
    pub fn set_reference(&mut self, image: &Frame, order: FaceOrder, index: usize) -> Result<()> {
        let mut faces = self.detector.detect(image)?;
        order.sort(&mut faces);
        let face = faces.get(index).ok_or(Error::NoReferenceFace { index, found: faces.len() })?;
        self.reference = Some(self.recognizer.embed(image, &face.landmarks)?);
        Ok(())
    }

    /// Swap faces regardless of identity again.
    pub fn clear_reference(&mut self) {
        self.reference = None;
    }

    fn largest_face_identity(&mut self, image: &Frame) -> Result<Embedding> {
        let face = self.detector.detect(image)?.into_iter().next().ok_or(Error::NoSourceFace)?;
        self.recognizer.embed(image, &face.landmarks)
    }

    /// Load every model `settings` needs, so the first frame isn't slow.
    pub fn prepare(&mut self, settings: &Settings) -> Result<()> {
        if self.swap_model != Some(settings.swap_model) {
            let mut current = async_swap::lock(&self.swapper);
            (*current, self.swap_model) = (None, None); // free VRAM before loading the replacement
            let swapper = Swapper::new(&self.engine, settings.swap_model)?;
            self.condition = self.identity.as_ref().map(|id| swapper.condition(id).into());
            (*current, self.swap_model) = (Some(swapper), Some(settings.swap_model));
            self.reuse.invalidate_all();
        }
        self.detector.prepare(settings.detector_size)?;
        self.detector.prepare(self.tracker.config.region_size)?;
        if settings.enhance && self.enhancer.is_none() {
            self.enhancer = Some(Enhancer::new(&self.engine)?);
        }
        self.masker.prepare(&self.engine, &settings.mask_spec())
    }

    /// Swap faces in `frame` in place. `live` enables temporal stabilisation.
    pub fn process(&mut self, frame: &mut Frame, settings: &Settings, live: bool) -> Result<Stats> {
        let started = Instant::now();
        if settings.mirror {
            frame.mirror();
        }
        self.prepare(settings)?;
        let mut stats = Stats::default();
        self.masker.begin_frame(&settings.mask_spec());
        if let Some(condition) = self.condition.clone() {
            let faces = self.detect(frame, settings, live)?;
            stats.faces = faces.len();
            stats.detect_ms = elapsed_ms(started);

            let swap_started = Instant::now();
            let size = pixel_boost::boosted_size(settings.swap_model.resolution(), settings.pixel_boost);
            if settings.enhance {
                self.restoration.original.clear();
                self.restoration.original.extend(faces.iter().map(|f| detail_region(frame, f).map_or(0.0, |r| detail(frame, r))));
            }
            if faces.len() != self.restoration.amounts.len() {
                self.restoration.amounts.clear();
            }
            let timewarp = live && timewarp_enabled();
            let budget = if live && !timewarp { settings.max_faces_per_frame } else { 0 };
            self.reuse.plan(&faces, budget, size, &mut self.fresh);
            if timewarp {
                self.timewarp(frame, &faces, &condition, size, &mut stats)?;
            }
            for (index, face) in faces.iter().enumerate() {
                let fresh = self.fresh[index];
                stats.swaps_reused += usize::from(!fresh);
                self.swap_face(index, frame, face, &condition, settings, &mut stats, live, fresh)?;
            }
            if !faces.is_empty() {
                let ages = (0..faces.len()).filter_map(|index| self.reuse.age(index)).map(|age| age.as_secs_f32() * 1000.0);
                stats.swap_age_ms = ages.sum::<f32>() / faces.len() as f32;
            }
            stats.swap_ms = elapsed_ms(swap_started);
            let masks = self.masker.stats();
            (stats.occlusion_ms, stats.parser_ms, stats.masks_computed, stats.masks_reused) =
                (masks.occlusion_ms, masks.parser_ms, masks.computed, masks.reused);

            if settings.enhance {
                let enhance_started = Instant::now();
                for (index, face) in faces.iter().enumerate() {
                    let Some(region) = detail_region(frame, face) else { continue };
                    let (original, swapped) = (self.restoration.original[index], detail(frame, region));
                    // About as detailed as the original: no restoration, once a live fade-out ends.
                    let settled = (swapped >= original * DETAIL_TOLERANCE).then(|| self.restoration.settle(index, faces.len(), 0.0, live));
                    if settled.is_some_and(|amount| amount < MIN_AMOUNT) {
                        stats.restorations_skipped += 1;
                        continue;
                    }
                    let crop = self.enhancer.as_mut().expect("prepared").enhance(frame, &face.landmarks)?;
                    let PasteMask { mask, resampled, realigned } = &mut self.paste_mask;
                    // Model masks live in the swap's ArcFace alignment; GFPGAN works in FFHQ's.
                    let model_mask = self.masker.mask(index).map(|m| {
                        let to_swap_space = Template::Ffhq512
                            .align(&face.landmarks, MASK_SIZE)
                            .invert()
                            .then(&Template::Arcface128.align(&face.landmarks, MASK_SIZE));
                        warp_plane_into(m, MASK_SIZE, &to_swap_space, MASK_SIZE, 1.0, realigned);
                        (realigned.as_slice(), MASK_SIZE)
                    });
                    let shape = self.shape_masks.get(Shape::Box, crop.size, settings.mask_blur, [0.0; 4]);
                    compose_into(shape, crop.size, model_mask, mask, resampled);
                    let amount = if let Some(amount) = settled {
                        amount
                    } else {
                        // Detail of the fully restored face, from a trial paste into a copy of the region.
                        let trial = &mut self.restoration.trial;
                        frame.crop_into(region, trial);
                        let to_crop = Affine([1.0, 0.0, region.x as f32, 0.0, 1.0, region.y as f32]).then(&crop.to_crop);
                        paste_back(trial, &crop.pixels, mask, crop.size, &to_crop, 1.0);
                        let restored = detail(trial, Rect::full(region.width, region.height));
                        let needed = if restored > swapped { ((original - swapped) / (restored - swapped)).clamp(0.0, 1.0) } else { 0.0 };
                        self.restoration.settle(index, faces.len(), needed, live)
                    };
                    paste(frame, &crop, mask, amount * settings.enhance_blend);
                }
                stats.enhance_ms = elapsed_ms(enhance_started);
            }
        }
        stats.total_ms = elapsed_ms(started);
        Ok(stats)
    }

    /// Async timewarp's part of a frame: land the swap that finished, mark the faces without a
    /// crop to be swapped right away (`fresh`), and hand the next face to the swap thread. Runs
    /// before anything is pasted, so the crop handed over is the camera's.
    fn timewarp(&mut self, frame: &Frame, faces: &[Face], condition: &Arc<[f32]>, size: usize, stats: &mut Stats) -> Result<()> {
        if self.async_swap.is_none() {
            self.async_swap = Some(AsyncSwap::spawn(self.swapper.clone())?);
        }
        let worker = self.async_swap.as_mut().expect("started above");
        if let Some(done) = worker.take() {
            let pixels = done.pixels?;
            if done.size == size {
                self.reuse.store_if_current(done.slot, done.generation, &pixels, &done.original, &done.to_crop, size, done.taken);
            }
            worker.recycle(done.original);
        }
        for (index, fresh) in self.fresh.iter_mut().enumerate() {
            *fresh = !self.reuse.is_valid(index);
        }
        if !worker.is_busy() && !faces.is_empty() {
            let next = self.reuse.next_turn(faces.len());
            // A face swapped synchronously this frame needn't go again, nor one that hasn't changed.
            let unchanged = self.reuse.change(next, frame).is_some_and(|change| change < SAME_FACE_CHANGE)
                && self.reuse.age(next).is_some_and(|age| age < MAX_SWAP_AGE);
            stats.swaps_skipped += usize::from(unchanged);
            if !self.fresh[next] && !unchanged {
                let to_crop = Template::Arcface128.align(&faces[next].landmarks, size);
                let mut crop = worker.buffer();
                crop.resize(size * size * Frame::CHANNELS, 0);
                warp_crop_into(frame, &to_crop, size, &mut crop);
                worker.submit(next, self.reuse.generation(next), condition, crop, to_crop, size);
            }
        }
        Ok(())
    }

    /// Swap face number `index` into `frame` — afresh, or with its last swap when not `fresh`;
    /// its model mask stays in the masker for the enhancer.
    #[expect(clippy::too_many_arguments, reason = "one call site; the arguments are the frame's context")]
    fn swap_face(
        &mut self,
        index: usize,
        frame: &mut Frame,
        target: &Face,
        condition: &[f32],
        settings: &Settings,
        stats: &mut Stats,
        live: bool,
        fresh: bool,
    ) -> Result<()> {
        let size = pixel_boost::boosted_size(settings.swap_model.resolution(), settings.pixel_boost);
        let Swapped { mut face, original } = if fresh {
            // Waits for a swap the swap thread has in flight.
            let mut swapper = async_swap::lock(&self.swapper);
            let swapped = pixel_boost::swap(swapper.as_mut().expect("prepared"), frame, &target.landmarks, condition, size)?;
            drop(swapper);
            self.reuse.store(index, &swapped.face.pixels, &swapped.original, &swapped.face.to_crop, size);
            swapped
        } else {
            // The last swap, pasted through the face's current alignment.
            let to_crop = Template::Arcface128.align(&target.landmarks, size);
            let mut pixels = std::mem::take(&mut self.reused_crop);
            self.reuse.load(index, &mut pixels);
            Swapped { face: AlignedCrop { pixels, size, to_crop }, original: warp_crop(frame, &to_crop, size) }
        };

        let mask_started = Instant::now();
        let model_mask = self.masker.face_mask(index, frame, target, &original, &face, live)?;
        let PasteMask { mask, resampled, .. } = &mut self.paste_mask;
        let shape = self.shape_masks.get(Shape::Face, size, settings.mask_blur, settings.mask_padding);
        compose_into(shape, size, model_mask.map(|m| (m, MASK_SIZE)), mask, resampled);
        match_color(&mut face.pixels, &original, mask, settings.color_correction);
        stats.mask_ms += elapsed_ms(mask_started);

        paste(frame, &face, mask, 1.0);
        if !fresh {
            self.reused_crop = face.pixels;
        }
        Ok(())
    }

    fn detect(&mut self, frame: &Frame, settings: &Settings, live: bool) -> Result<Vec<Face>> {
        self.detector.input_size = settings.detector_size;
        let faces = if live {
            let mut source = TrackingDetector { detector: &mut self.detector, region_size: self.tracker.config.region_size };
            self.tracker.detect(&mut source, frame)?.0
        } else {
            self.detector.detect(frame)?
        };
        let similarities = match &self.reference {
            Some(reference) => Some(
                faces.iter().map(|f| Ok(self.recognizer.embed(frame, &f.landmarks)?.similarity(reference))).collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };
        let mut faces = select(faces, similarities.as_deref(), 1.0 - settings.reference_distance, settings.face_order, settings.many_faces);
        // Follow the raw (unsmoothed) positions of the faces being swapped.
        self.tracker.track(if live { &faces } else { &[] });
        if live && settings.stabilize {
            self.flow.refine(frame, &mut faces);
        } else {
            self.flow.reset();
        }
        Ok(faces)
    }
}

fn paste(frame: &mut Frame, crop: &AlignedCrop, mask: &[f32], opacity: f32) {
    paste_back(frame, &crop.pixels, mask, crop.size, &crop.to_crop, opacity);
}

fn elapsed_ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}
