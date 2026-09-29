use super::pixel_boost::{self, Swapped};
use super::tracker::{FaceTracker, TrackingDetector};
use super::{FaceOrder, FaceSmoother, Settings, Stats, select};
use crate::face::masker::{MASK_SIZE, compose_into};
use crate::face::{AlignedCrop, Detector, Embedding, Enhancer, Face, Masker, Recognizer, Swapper};
use crate::image::{Frame, Padding, box_mask, match_color, paste_back};
use crate::runtime::{Backend, Engine};
use crate::{Error, Result};
use std::collections::VecDeque;
use std::time::Instant;

type MaskKey = (usize, u32, [u32; 4]);
/// Box masks kept: enough for the swap and enhance crops, small enough that dragging a
/// blur/padding slider can't grow the cache without bound.
const MASK_CACHE: usize = 4;

/// Recently used box masks, most recent first.
struct BoxMasks(VecDeque<(MaskKey, Vec<f32>)>);

impl BoxMasks {
    fn get(&mut self, size: usize, blur: f32, padding: Padding) -> &[f32] {
        let key = (size, blur.to_bits(), padding.map(f32::to_bits));
        let entry = if let Some(i) = self.0.iter().position(|(k, _)| *k == key) {
            self.0.remove(i).expect("index in range")
        } else {
            self.0.truncate(MASK_CACHE - 1);
            (key, box_mask(size, blur, padding))
        };
        self.0.push_front(entry);
        &self.0[0].1
    }
}

/// Buffers for the final paste mask, reused every frame.
#[derive(Default)]
struct PasteMask {
    mask: Vec<f32>,
    resampled: Vec<f32>,
}

/// Owns the models and per-stream state; call [`Processor::process`] once per frame.
pub struct Processor {
    engine: Engine,
    detector: Detector,
    recognizer: Recognizer,
    swapper: Option<Swapper>,
    enhancer: Option<Enhancer>,
    masker: Masker,
    identity: Option<Embedding>,
    /// `identity` converted for the current swapper.
    condition: Option<Vec<f32>>,
    /// Only faces like this one are swapped, when set.
    reference: Option<Embedding>,
    smoother: FaceSmoother,
    tracker: FaceTracker,
    box_masks: BoxMasks,
    paste_mask: PasteMask,
    last_frame: Option<Instant>,
}

impl Processor {
    pub fn new(engine: Engine) -> Result<Self> {
        Ok(Self {
            detector: Detector::new(&engine)?,
            recognizer: Recognizer::new(&engine)?,
            engine,
            swapper: None,
            enhancer: None,
            masker: Masker::default(),
            identity: None,
            condition: None,
            reference: None,
            smoother: FaceSmoother::default(),
            tracker: FaceTracker::default(),
            box_masks: BoxMasks(VecDeque::with_capacity(MASK_CACHE)),
            paste_mask: PasteMask::default(),
            last_frame: None,
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
        self.condition = self.swapper.as_ref().map(|s| s.condition(&identity));
        self.identity = Some(identity);
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
        if self.swapper.as_ref().map(Swapper::kind) != Some(settings.swap_model) {
            self.swapper = None; // free VRAM before loading the replacement
            let swapper = Swapper::new(&self.engine, settings.swap_model)?;
            self.condition = self.identity.as_ref().map(|id| swapper.condition(id));
            self.swapper = Some(swapper);
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
            for (index, face) in faces.iter().enumerate() {
                self.swap_face(index, frame, face, &condition, settings, &mut stats, live)?;
            }
            stats.swap_ms = elapsed_ms(swap_started);
            let masks = self.masker.stats();
            (stats.occlusion_ms, stats.parser_ms, stats.masks_computed, stats.masks_reused) =
                (masks.occlusion_ms, masks.parser_ms, masks.computed, masks.reused);

            if settings.enhance {
                let enhance_started = Instant::now();
                for (index, face) in faces.iter().enumerate() {
                    let crop = self.enhancer.as_mut().expect("prepared").enhance(frame, &face.landmarks)?;
                    let PasteMask { mask, resampled } = &mut self.paste_mask;
                    let box_mask = self.box_masks.get(crop.size, settings.mask_blur, [0.0; 4]);
                    compose_into(box_mask, crop.size, self.masker.mask(index).map(|m| (m, MASK_SIZE)), mask, resampled);
                    paste(frame, &crop, mask, settings.enhance_blend);
                }
                stats.enhance_ms = elapsed_ms(enhance_started);
            }
        }
        stats.total_ms = elapsed_ms(started);
        Ok(stats)
    }

    /// Swap face number `index` into `frame`; its model mask stays in the masker for the enhancer.
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
    ) -> Result<()> {
        let swapper = self.swapper.as_mut().expect("prepared");
        let size = pixel_boost::boosted_size(swapper.kind().resolution(), settings.pixel_boost);
        let Swapped { mut face, original } = pixel_boost::swap(swapper, frame, &target.landmarks, condition, size)?;

        let mask_started = Instant::now();
        let model_mask = self.masker.face_mask(index, frame, target, &original, &face, live)?;
        let PasteMask { mask, resampled } = &mut self.paste_mask;
        let box_mask = self.box_masks.get(size, settings.mask_blur, settings.mask_padding);
        compose_into(box_mask, size, model_mask.map(|m| (m, MASK_SIZE)), mask, resampled);
        match_color(&mut face.pixels, &original, mask, settings.color_correction);
        stats.mask_ms += elapsed_ms(mask_started);

        paste(frame, &face, mask, 1.0);
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
        let now = Instant::now();
        let dt = self.last_frame.map_or(1.0 / 30.0, |t| (now - t).as_secs_f32());
        self.last_frame = Some(now);
        if live && settings.stabilize {
            self.smoother.smooth(&mut faces, dt);
        } else {
            self.smoother.reset();
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
