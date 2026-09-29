use super::processor;
use crate::cli::Options;
use crate::jobs::video::{Progress, Rendered, VideoJob, render};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use wajh_media::{Encoder, probe};

/// Frames between progress lines.
const REPORT_EVERY: u64 = 30;

pub fn run(options: &Options, source: &[PathBuf], target: &str, output: &str, encoder: Option<String>) -> Result<()> {
    let info = probe(target)?;
    let encoder = encoder.map_or_else(Encoder::best_available, Encoder);
    eprintln!("{}x{} @ {:.2} fps, encoder {}", info.width, info.height, info.fps, encoder.0);
    let mut processor = processor(options, source)?;
    let job = VideoJob { target, output, encoder: &encoder, info: &info };
    let never = AtomicBool::new(false);
    let rendered = render(&mut processor, &options.settings(), &job, &never, &mut |progress, _| {
        if progress.done.is_multiple_of(REPORT_EVERY) {
            report_progress(progress);
        }
    })?;
    if let Rendered::Complete(progress) = rendered {
        eprintln!("\ndone: {} frames in {:.1}s → {output}", progress.done, progress.elapsed.as_secs_f64());
    }
    Ok(())
}

fn report_progress(progress: &Progress) {
    let (done, fps) = (progress.done, progress.fps());
    match progress.total {
        Some(total) => eprint!("\r{done}/{total} frames, {fps:.1} fps   "),
        None => eprint!("\r{done} frames, {fps:.1} fps   "),
    }
}
