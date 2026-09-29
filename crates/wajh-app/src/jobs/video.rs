//! Decode → swap → encode for a whole video file, shared by `wajh video` and the Video screen.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use wajh_core::{Frame, Processor, Settings};
use wajh_media::{Encoder, VideoInfo, VideoReader};

/// Frames buffered between the decoder, the swapper and the encoder: enough to ride out
/// hiccups, small enough that memory stays flat at 4K.
const QUEUE: usize = 8;

/// What to render.
#[derive(Clone, Debug)]
pub struct VideoJob<'a> {
    pub target: &'a str,
    pub output: &'a str,
    pub encoder: &'a Encoder,
    pub info: &'a VideoInfo,
}

/// How far a render has got.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Progress {
    pub done: u64,
    /// Expected frame count, when the container reports it (or its duration).
    pub total: Option<u64>,
    pub elapsed: Duration,
}

impl Progress {
    /// Average frames per second so far.
    #[must_use]
    pub fn fps(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds > 0.0 { self.done as f64 / seconds } else { 0.0 }
    }

    /// Completed fraction in `0..=1`, when the total is known.
    #[must_use]
    pub fn fraction(&self) -> Option<f32> {
        self.total.filter(|&t| t > 0).map(|total| (self.done as f64 / total as f64).min(1.0) as f32)
    }

    /// Estimated time left at the current average rate.
    #[must_use]
    pub fn eta(&self) -> Option<Duration> {
        let (total, fps) = (self.total?, self.fps());
        (fps > 0.0).then(|| Duration::from_secs_f64(total.saturating_sub(self.done) as f64 / fps))
    }
}

/// How a render ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rendered {
    Complete(Progress),
    /// Stopped through the cancel flag; the partial output file was removed.
    Cancelled,
}

/// Decode → swap → encode, each on its own thread, connected by bounded channels so ffmpeg
/// never waits on inference and memory stays flat.
///
/// `on_frame` sees every processed frame (before it is encoded) with the progress so far;
/// keep it cheap. Setting `cancel` stops the render at the next frame and deletes the output.
pub fn render(
    processor: &mut Processor,
    settings: &Settings,
    job: &VideoJob<'_>,
    cancel: &AtomicBool,
    on_frame: &mut dyn FnMut(&Progress, &Frame),
) -> Result<Rendered> {
    let reader = VideoReader::open(job.target, job.info)?;
    let mut sink = job.encoder.open(job.output, job.target, job.info)?;
    let started = Instant::now();
    let mut progress = Progress { total: job.info.total_frames(), ..Progress::default() };

    let (decoded_tx, decoded) = crossbeam_channel::bounded::<Frame>(QUEUE);
    let (swapped, swapped_rx) = crossbeam_channel::bounded::<Frame>(QUEUE);
    let (swap_result, encode_result) = std::thread::scope(|scope| {
        scope.spawn(move || reader.map(|frame| decoded_tx.send(frame)).take_while(Result::is_ok).count());
        let encoder = scope.spawn(move || -> wajh_media::Result<()> {
            for frame in swapped_rx {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(()); // dropping the sink kills ffmpeg
                }
                sink.write(&frame)?;
            }
            if cancel.load(Ordering::Relaxed) { Ok(()) } else { sink.finish() }
        });

        let swap_result = (|| -> Result<()> {
            for mut frame in decoded {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                processor.process(&mut frame, settings, false)?;
                progress.done += 1;
                progress.elapsed = started.elapsed();
                on_frame(&progress, &frame);
                if swapped.send(frame).is_err() {
                    break; // the encoder failed; its error is reported below
                }
            }
            Ok(())
        })();
        drop(swapped);
        (swap_result, encoder.join())
    });

    let finished = swap_result.and_then(|()| encode_result.ok().context("encoder thread panicked")?.context("encoding failed"));
    if cancel.load(Ordering::Relaxed) || finished.is_err() {
        let _ = std::fs::remove_file(job.output);
    }
    finished?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(Rendered::Cancelled);
    }
    progress.elapsed = started.elapsed();
    Ok(Rendered::Complete(progress))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_estimates() {
        let progress = Progress { done: 50, total: Some(200), elapsed: Duration::from_secs(5) };
        assert_eq!(progress.fps(), 10.0);
        assert_eq!(progress.fraction(), Some(0.25));
        assert_eq!(progress.eta(), Some(Duration::from_secs(15)));
    }

    #[test]
    fn progress_without_total_or_time() {
        let progress = Progress::default();
        assert_eq!(progress.fps(), 0.0);
        assert_eq!(progress.fraction(), None);
        assert_eq!(progress.eta(), None);
        let overrun = Progress { done: 12, total: Some(10), elapsed: Duration::from_secs(1) };
        assert_eq!(overrun.fraction(), Some(1.0));
        assert_eq!(overrun.eta(), Some(Duration::ZERO));
    }
}
