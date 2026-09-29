use super::processor;
use crate::cli::Options;
use anyhow::Result;
use std::path::{Path, PathBuf};
use wajh_core::{Frame, Stats};

pub fn run(options: &Options, source: &[PathBuf], target: &Path, iterations: usize) -> Result<()> {
    let mut processor = processor(options, source)?;
    let settings = options.settings();
    let image = Frame::open(target)?;
    // Warm-up: first runs pay for cuDNN autotuning / TensorRT engine builds.
    for _ in 0..3 {
        processor.process(&mut image.clone(), &settings, true)?;
    }
    let mut total = Stats::default();
    for _ in 0..iterations {
        total += processor.process(&mut image.clone(), &settings, true)?;
    }
    println!("{}x{} on {}: {}", image.width, image.height, processor.backend(), total / iterations.max(1) as f32);
    Ok(())
}
