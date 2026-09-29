use super::image::{THUMBNAIL_MAX, swap, thumbnail};
use super::video::{Rendered, VideoJob, render};
use super::{Finished, Job, JobShared, Queued, Status};
use crate::live::EngineOptions;
use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use wajh_core::{Engine, Frame, Processor, Settings};

/// How often a rendering video refreshes the UI (thumbnail and progress).
const PREVIEW_INTERVAL: Duration = Duration::from_millis(200);

/// The processor and the source face it currently holds.
struct Loaded {
    processor: Processor,
    source: Option<PathBuf>,
}

pub(super) fn run(engine: &EngineOptions, inbox: &Receiver<Queued>, repaint: &dyn Fn()) {
    let mut loaded: Option<Loaded> = None;
    for Queued { job, shared } in inbox {
        let set = |status: Status| {
            *shared.status.lock() = status;
            repaint();
        };
        if shared.cancel.load(Ordering::Relaxed) {
            set(Status::Cancelled);
            continue;
        }
        set(Status::Loading);
        let status = match execute(engine, &mut loaded, job, &shared, repaint) {
            Ok(status) => status,
            Err(error) => Status::Failed(format!("{error:#}")),
        };
        set(status);
    }
}

fn execute(engine: &EngineOptions, loaded: &mut Option<Loaded>, job: Job, shared: &JobShared, repaint: &dyn Fn()) -> Result<Status> {
    let (source, settings) = match &job {
        Job::Image { source, settings, .. } | Job::Video { source, settings, .. } => (source, offline(settings)),
    };
    let processor = prepare(engine, loaded, source, &settings)?;
    match job {
        Job::Image { target, .. } => {
            let swapped = swap(processor, &settings, &target)?;
            let stats = swapped.stats;
            *shared.image.lock() = Some(swapped);
            Ok(Status::Done(Finished::Image(stats)))
        }
        Job::Video { target, output, encoder, info, .. } => {
            let (target_str, output_str) = (target.to_string_lossy(), output.to_string_lossy());
            let job = VideoJob { target: &target_str, output: &output_str, encoder: &encoder, info: &info };
            let mut last_update: Option<Instant> = None;
            let rendered = render(processor, &settings, &job, &shared.cancel, &mut |progress, frame| {
                *shared.status.lock() = Status::Running(*progress);
                if last_update.is_none_or(|t| t.elapsed() >= PREVIEW_INTERVAL) {
                    last_update = Some(Instant::now());
                    *shared.preview.lock() = Some(thumbnail(frame, THUMBNAIL_MAX));
                    repaint();
                }
            })?;
            Ok(match rendered {
                Rendered::Complete(progress) => Status::Done(Finished::Video { output, progress }),
                Rendered::Cancelled => Status::Cancelled,
            })
        }
    }
}

/// Live-only options off: files are neither mirrored nor stabilised.
fn offline(settings: &Settings) -> Settings {
    Settings { mirror: false, stabilize: false, ..settings.clone() }
}

/// Load the engine on first use, switch the source face when it changed and load the models.
fn prepare<'a>(engine: &EngineOptions, loaded: &'a mut Option<Loaded>, source: &Path, settings: &Settings) -> Result<&'a mut Processor> {
    if loaded.is_none() {
        let engine = Engine::init(engine.runtime.as_deref(), engine.device)?;
        *loaded = Some(Loaded { processor: Processor::new(engine)?, source: None });
    }
    let state = loaded.as_mut().expect("loaded above");
    if state.source.as_deref() != Some(source) {
        state.source = None;
        let frame = Frame::open(source).with_context(|| format!("could not open {}", source.display()))?;
        state.processor.set_source(&frame)?;
        state.source = Some(source.to_path_buf());
    }
    state.processor.prepare(settings)?;
    Ok(&mut state.processor)
}
