use super::processor;
use crate::cli::{CameraArgs, Options};
use anyhow::{Result, bail};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wessh_core::{Settings, Stats};
use wessh_media::{Camera, FrameSink, LatestFrame, OutputTarget};

pub fn run(options: &Options, source: &[PathBuf], camera: &CameraArgs, out: Option<&str>) -> Result<()> {
    let mut processor = processor(options, source)?;
    let settings = Settings { mirror: camera.mirror, ..options.settings() };
    let mailbox = Arc::new(LatestFrame::default());
    let mut camera_capture = Camera::start(&camera.config(), mailbox.clone())?;
    let mut sink = out.map(|t| FrameSink::live(&OutputTarget::parse(t), camera.width, camera.height, camera.fps)).transpose()?;

    eprintln!("running — Ctrl+C to stop");
    let (mut seen, mut window, mut frames, mut total) = (0, Instant::now(), 0u32, Stats::default());
    loop {
        let Some(mut frame) = mailbox.take_newer(&mut seen, Duration::from_secs(2)) else {
            let reason = camera_capture.failure().unwrap_or_else(|| "the camera stopped delivering frames".to_owned());
            bail!("{reason}");
        };
        total += processor.process(&mut frame, &settings, true)?;
        if let Some(sink) = sink.as_mut() {
            sink.write(&frame)?;
        }
        mailbox.recycle(frame);
        frames += 1;
        if window.elapsed() >= Duration::from_secs(1) {
            let fps = frames as f32 / window.elapsed().as_secs_f32();
            eprint!("\r{fps:.1} fps | {}   ", total / frames as f32);
            (window, frames, total) = (Instant::now(), 0, Stats::default());
        }
    }
}
