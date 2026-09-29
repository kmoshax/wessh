use super::tracker::{FaceTracker, TrackingDetector};
use super::{FaceSmoother, Settings, Stats};
use crate::face::{AlignedCrop, Detector, Embedding, Enhancer, Face, Recognizer, Swapper};
use crate::image::{Frame, Padding, box_mask, paste_back};
use crate::runtime::{Backend, Engine};
use crate::{Error, Result};
use std::collections::HashMap;
use std::time::Instant;

type MaskKey = (usize, u32, [u32; 4]);

/// Owns the models and per-stream state; call [`Processor::process`] once per frame.
pub struct Processor {
    engine: Engine,
    detector: Detector,
    recognizer: Recognizer,
    swapper: Option<Swapper>,
    enhancer: Option<Enhancer>,
    identity: Option<Embedding>,
    /// `identity` converted for the current swapper.
    condition: Option<Vec<f32>>,
    smoother: FaceSmoother,
    tracker: FaceTracker,
    masks: HashMap<MaskKey, Vec<f32>>,
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
            identity: None,
            condition: None,
            smoother: FaceSmoother::default(),
            tracker: FaceTracker::default(),
            masks: HashMap::new(),
            last_frame: None,
        })
    }

    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.engine.backend()
    }

    /// Use the largest face in `image` as the identity to swap in.
    pub fn set_source(&mut self, image: &Frame) -> Result<()> {
        let face = self.detector.detect(image)?.into_iter().next().ok_or(Error::NoSourceFace)?;
        let identity = self.recognizer.embed(image, &face.landmarks)?;
        self.condition = self.swapper.as_ref().map(|s| s.condition(&identity));
        self.identity = Some(identity);
        Ok(())
    }

    #[must_use]
    pub const fn has_source(&self) -> bool {
        self.identity.is_some()
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
        Ok(())
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
            for face in &faces {
                let crop = self.swapper.as_mut().expect("prepared").swap(frame, &face.landmarks, &condition)?;
                self.paste(frame, &crop, settings.mask_blur, settings.mask_padding, 1.0);
            }
            stats.swap_ms = elapsed_ms(swap_started);

            if settings.enhance {
                let enhance_started = Instant::now();
                for face in &faces {
                    let crop = self.enhancer.as_mut().expect("prepared").enhance(frame, &face.landmarks)?;
                    self.paste(frame, &crop, settings.mask_blur, [0.0; 4], settings.enhance_blend);
                }
                stats.enhance_ms = elapsed_ms(enhance_started);
            }
        }
        stats.total_ms = elapsed_ms(started);
        Ok(stats)
    }

    fn detect(&mut self, frame: &Frame, settings: &Settings, live: bool) -> Result<Vec<Face>> {
        self.detector.input_size = settings.detector_size;
        let limit = if settings.many_faces { usize::MAX } else { 1 };
        let mut faces = if live {
            let mut source = TrackingDetector { detector: &mut self.detector, region_size: self.tracker.config.region_size };
            self.tracker.detect(&mut source, frame, limit)?.0
        } else {
            self.tracker.reset();
            let mut faces = self.detector.detect(frame)?;
            faces.truncate(limit);
            faces
        };
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

    fn paste(&mut self, frame: &mut Frame, crop: &AlignedCrop, blur: f32, padding: Padding, opacity: f32) {
        let key = (crop.size, blur.to_bits(), padding.map(f32::to_bits));
        let mask = self.masks.entry(key).or_insert_with(|| box_mask(crop.size, blur, padding));
        paste_back(frame, &crop.pixels, mask, crop.size, &crop.to_crop, opacity);
    }
}

fn elapsed_ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}
