use super::processor;
use crate::cli::Options;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;
use wajh_core::Frame;
use wajh_media::{Encoder, VideoReader, probe};

/// Decode → swap → encode, each on its own thread, connected by bounded channels so ffmpeg
/// never waits on inference and memory stays flat.
pub fn run(options: &Options, source: &Path, target: &str, output: &str, encoder: Option<String>) -> Result<()> {
    const QUEUE: usize = 8;
    let info = probe(target)?;
    let encoder = encoder.map_or_else(Encoder::best_available, Encoder);
    eprintln!("{}x{} @ {:.2} fps, encoder {}", info.width, info.height, info.fps, encoder.0);
    let mut processor = processor(options, source)?;
    let settings = options.settings();

    let (decoded_tx, decoded) = crossbeam_channel::bounded::<Frame>(QUEUE);
    let (swapped, swapped_rx) = crossbeam_channel::bounded::<Frame>(QUEUE);
    let reader = VideoReader::open(target, &info)?;
    let decoder = std::thread::spawn(move || reader.map(|frame| decoded_tx.send(frame)).take_while(Result::is_ok).count());
    let mut sink = encoder.open(output, target, &info)?;
    let encoder = std::thread::spawn(move || -> wajh_media::Result<()> {
        for frame in swapped_rx {
            sink.write(&frame)?;
        }
        sink.finish()
    });

    let started = Instant::now();
    let mut done = 0u64;
    for mut frame in decoded {
        processor.process(&mut frame, &settings, false)?;
        done += 1;
        if done.is_multiple_of(30) {
            report_progress(done, info.frame_count, started);
        }
        if swapped.send(frame).is_err() {
            break;
        }
    }
    drop(swapped);
    decoder.join().ok().context("decoder thread panicked")?;
    encoder.join().ok().context("encoder thread panicked")??;
    eprintln!("\ndone: {done} frames in {:.1}s → {output}", started.elapsed().as_secs_f64());
    Ok(())
}

fn report_progress(done: u64, total: Option<u64>, started: Instant) {
    let fps = done as f64 / started.elapsed().as_secs_f64();
    match total {
        Some(total) => eprint!("\r{done}/{total} frames, {fps:.1} fps   "),
        None => eprint!("\r{done} frames, {fps:.1} fps   "),
    }
}
