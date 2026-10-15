use super::processor;
use crate::cli::{CameraArgs, Options};
use anyhow::{Result, bail};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wessh_core::{Settings, Stats};
use wessh_media::{Camera, FrameSink, LatestFrame, OutputTarget};

pub fn run(options: &Options, source: &[PathBuf], camera: &CameraArgs, out: Option<&str>) -> Result<()> {
    if Camera::needs_ffmpeg(&camera.camera) || out.is_some_and(|t| OutputTarget::parse(t).needs_ffmpeg()) {
        super::setup::ensure_ffmpeg()?;
    }
    let mut processor = processor(options, source)?;
    let settings = Settings { mirror: camera.mirror, ..options.settings() };
    let mailbox = Arc::new(LatestFrame::default());
    let mut camera_capture = Camera::start(&camera.config(), mailbox.clone())?;
    let mut sink = out.map(|t| FrameSink::live(&OutputTarget::parse(t), camera.width, camera.height, camera.fps)).transpose()?;

    eprintln!("running — Ctrl+C to stop");
    let (mut seen, mut window, mut frames, mut total) = (0, Instant::now(), 0u32, Stats::default());
    let (mut window_seen, mut latency) = (0, 0.0f32);
    loop {
        let Some(mut frame) = mailbox.take_newer(&mut seen, Duration::from_secs(2)) else {
            let reason = camera_capture.failure().unwrap_or_else(|| "the camera stopped delivering frames".to_owned());
            bail!("{reason}");
        };
        total += processor.process(&mut frame, &settings, true)?;
        if let Some(sink) = sink.as_mut() {
            sink.write(&frame)?;
        }
        latency += frame.captured.map_or(0.0, |captured| captured.elapsed().as_secs_f32() * 1000.0);
        mailbox.recycle(frame);
        frames += 1;
        if window.elapsed() >= Duration::from_secs(1) {
            let seconds = window.elapsed().as_secs_f32();
            let (fps, camera_fps) = (frames as f32 / seconds, (seen - window_seen) as f32 / seconds);
            let latency_ms = latency / frames as f32;
            eprint!("\r{fps:.1} fps (camera {camera_fps:.1}) | latency {latency_ms:.1} ms | {}   ", total / frames as f32);
            (window, frames, total, window_seen, latency) = (Instant::now(), 0, Stats::default(), seen, 0.0);
        }
    }
}
