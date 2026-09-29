use super::pixel_boost::{self, Swapped};
use super::tracker::{FaceTracker, TrackingDetector};
use super::{FaceOrder, FaceSmoother, Settings, Stats, select};
use crate::face::masker::{MASK_SIZE, compose};
use crate::face::{AlignedCrop, Detector, Embedding, Enhancer, Face, Masker, Recognizer, Swapper};
use crate::geometry::Landmarks5;
use crate::image::{Frame, Padding, box_mask, match_color, paste_back};
use crate::runtime::{Backend, Engine};
use crate::{Error, Result};
use std::collections::VecDeque;
use std::time::Instant;

type MaskKey = (usize, u32, [u32; 4]);
/// Box masks kept: enough for the swap and enhance crops, small enough that dragging a
/// blur/padding slider can't grow the cache without bound.
const MASK_CACHE: usize = 4;

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
    /// Most recently used first.
    masks: VecDeque<(MaskKey, Vec<f32>)>,
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
            masks: VecDeque::with_capacity(MASK_CACHE),
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
        if let Some(condition) = self.condition.clone() {
            let faces = self.detect(frame, settings, live)?;
            stats.faces = faces.len();
            stats.detect_ms = elapsed_ms(started);

            let swap_started = Instant::now();
            let mut model_masks = Vec::with_capacity(faces.len());
            for face in &faces {
                model_masks.push(self.swap_face(frame, &face.landmarks, &condition, settings, &mut stats)?);
            }
            stats.swap_ms = elapsed_ms(swap_started);

            if settings.enhance {
                let enhance_started = Instant::now();
                for (face, model_mask) in faces.iter().zip(&model_masks) {
                    let crop = self.enhancer.as_mut().expect("prepared").enhance(frame, &face.landmarks)?;
                    let mask = compose(
                        self.box_mask(crop.size, settings.mask_blur, [0.0; 4]),
                        crop.size,
                        model_mask.as_deref().map(|m| (m, MASK_SIZE)),
                    );
                    paste(frame, &crop, &mask, settings.enhance_blend);
                }
                stats.enhance_ms = elapsed_ms(enhance_started);
            }
        }
        stats.total_ms = elapsed_ms(started);
        Ok(stats)
    }

    /// Swap one face into `frame`; returns its model mask (for the enhancer to reuse).
    fn swap_face(
        &mut self,
        frame: &mut Frame,
        landmarks: &Landmarks5,
        condition: &[f32],
        settings: &Settings,
        stats: &mut Stats,
    ) -> Result<Option<Vec<f32>>> {
        let swapper = self.swapper.as_mut().expect("prepared");
        let size = pixel_boost::boosted_size(swapper.kind().resolution(), settings.pixel_boost);
        let Swapped { mut face, original } = pixel_boost::swap(swapper, frame, landmarks, condition, size)?;

        let mask_started = Instant::now();
        let model_mask = self.masker.face_mask(frame, landmarks, &face, &settings.mask_spec())?;
        let mask =
            compose(self.box_mask(size, settings.mask_blur, settings.mask_padding), size, model_mask.as_deref().map(|m| (m, MASK_SIZE)));
        match_color(&mut face.pixels, &original, &mask, settings.color_correction);
        stats.mask_ms += elapsed_ms(mask_started);

        paste(frame, &face, &mask, 1.0);
        Ok(model_mask)
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

    fn box_mask(&mut self, size: usize, blur: f32, padding: Padding) -> &[f32] {
        let key = (size, blur.to_bits(), padding.map(f32::to_bits));
        let entry = if let Some(i) = self.masks.iter().position(|(k, _)| *k == key) {
            self.masks.remove(i).expect("index in range")
        } else {
            self.masks.truncate(MASK_CACHE - 1);
            (key, box_mask(size, blur, padding))
        };
        self.masks.push_front(entry);
        &self.masks[0].1
    }
}

fn paste(frame: &mut Frame, crop: &AlignedCrop, mask: &[f32], opacity: f32) {
    paste_back(frame, &crop.pixels, mask, crop.size, &crop.to_crop, opacity);
}

fn elapsed_ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}
