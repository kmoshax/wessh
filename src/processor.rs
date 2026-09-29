//! The per-frame face swap pipeline.

use crate::engine::Engine;
use crate::face::{Detector, Enhancer, Recognizer, Swapper};
use crate::imgops::{self, Frame};
use crate::models::SwapModel;
use crate::smoothing::FaceSmoother;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Settings {
    pub swap_model: SwapModel,
    pub enhance: bool,
    /// 0..1, how much of the enhanced face is blended in.
    pub enhance_blend: f32,
    pub many_faces: bool,
    /// Landmark stabilisation for live video.
    pub smooth: bool,
    pub mask_blur: f32,
    /// top, right, bottom, left in percent of the crop.
    pub mask_padding: [f32; 4],
    pub mirror: bool,
    pub detector_size: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            swap_model: SwapModel::Inswapper,
            enhance: false,
            enhance_blend: 0.8,
            many_faces: false,
            smooth: true,
            mask_blur: 0.3,
            mask_padding: [0.0; 4],
            mirror: false,
            detector_size: 640,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub faces: usize,
    pub detect_ms: f32,
    pub swap_ms: f32,
    pub enhance_ms: f32,
    pub total_ms: f32,
}

pub struct Processor {
    engine: Engine,
    detector: Detector,
    recognizer: Recognizer,
    swapper: Option<Swapper>,
    enhancer: Option<Enhancer>,
    source_embedding: Option<Vec<f32>>,
    source_input: Option<Vec<f32>>,
    smoother: FaceSmoother,
    masks: HashMap<(usize, u32, [u32; 4]), Vec<f32>>,
    last_frame: Option<Instant>,
}

impl Processor {
    pub fn new(engine: Engine) -> Result<Self> {
        let detector = Detector::new(&engine)?;
        let recognizer = Recognizer::new(&engine)?;
        Ok(Self {
            engine,
            detector,
            recognizer,
            swapper: None,
            enhancer: None,
            source_embedding: None,
            source_input: None,
            smoother: FaceSmoother::new(1.2, 8.0),
            masks: HashMap::new(),
            last_frame: None,
        })
    }

    pub fn backend(&self) -> &'static str {
        self.engine.backend
    }

    /// Detect the largest face in `image` and use it as the identity to swap in.
    pub fn set_source(&mut self, image: &Frame) -> Result<()> {
        let faces = self.detector.detect(image)?;
        let face = faces.first().context("no face found in the source image")?;
        let emb = self.recognizer.embed(image, &face.kps)?;
        self.source_input = self.swapper.as_ref().map(|s| s.prepare_source(&emb));
        self.source_embedding = Some(emb);
        Ok(())
    }

    fn ensure_models(&mut self, settings: &Settings) -> Result<()> {
        if self.swapper.as_ref().map(|s| s.kind) != Some(settings.swap_model) {
            self.swapper = None;
            let swapper = Swapper::new(&self.engine, settings.swap_model)?;
            self.source_input = self.source_embedding.as_ref().map(|e| swapper.prepare_source(e));
            self.swapper = Some(swapper);
        }
        if settings.enhance && self.enhancer.is_none() {
            self.enhancer = Some(Enhancer::new(&self.engine)?);
        }
        Ok(())
    }

    /// Load everything `settings` needs up front (so the first frame isn't slow).
    pub fn prepare(&mut self, settings: &Settings) -> Result<()> {
        self.ensure_models(settings)
    }

    fn mask(&mut self, size: usize, blur: f32, padding: [f32; 4]) -> &[f32] {
        let key = (size, blur.to_bits(), padding.map(f32::to_bits));
        self.masks.entry(key).or_insert_with(|| imgops::box_mask(size, blur, padding))
    }

    /// Swap faces in `frame` in place. `live` enables temporal smoothing.
    pub fn process(&mut self, frame: &mut Frame, settings: &Settings, live: bool) -> Result<Stats> {
        let t0 = Instant::now();
        let mut stats = Stats::default();
        if settings.mirror {
            frame.mirror();
        }
        self.ensure_models(settings)?;
        self.detector.input_size = settings.detector_size;
        let Some(source) = self.source_input.clone() else {
            stats.total_ms = ms(t0);
            return Ok(stats);
        };

        let mut faces = self.detector.detect(frame)?;
        if !settings.many_faces {
            faces.truncate(1);
        }
        stats.detect_ms = ms(t0);
        stats.faces = faces.len();

        let now = Instant::now();
        let dt = self.last_frame.map(|t| (now - t).as_secs_f32()).unwrap_or(1.0 / 30.0);
        self.last_frame = Some(now);
        if live && settings.smooth {
            let mut lm: Vec<_> = faces.iter().map(|f| (f.kps, f.size())).collect();
            self.smoother.smooth(&mut lm, dt);
            for (f, (k, _)) in faces.iter_mut().zip(lm) {
                f.kps = k;
            }
        } else {
            self.smoother.reset();
        }

        let t1 = Instant::now();
        for face in &faces {
            let swapper = self.swapper.as_mut().unwrap();
            let size = swapper.size;
            let (crop, m) = swapper.swap(frame, &face.kps, &source)?;
            let mask = self.mask(size, settings.mask_blur, settings.mask_padding).to_vec();
            imgops::paste_back(frame, &crop, &mask, size, &m, 1.0);
        }
        stats.swap_ms = ms(t1);

        if settings.enhance {
            let t2 = Instant::now();
            for face in &faces {
                let (crop, m) = self.enhancer.as_mut().unwrap().enhance(frame, &face.kps)?;
                let mask = self.mask(Enhancer::SIZE, settings.mask_blur, [0.0; 4]).to_vec();
                imgops::paste_back(frame, &crop, &mask, Enhancer::SIZE, &m, settings.enhance_blend);
            }
            stats.enhance_ms = ms(t2);
        }
        stats.total_ms = ms(t0);
        Ok(stats)
    }
}

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}
