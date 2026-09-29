//! `wajh bench`: end-to-end per-stage latency, plus an optional CPU-kernel breakdown.

use super::processor;
use crate::cli::Options;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use wajh_core::geometry::Template;
use wajh_core::image::{box_mask, letterbox_nchw, paste_back, warp_crop};
use wajh_core::{Frame, Stats};

/// Iterations that pay for cuDNN autotuning, TensorRT engine builds and CUDA graph capture.
const WARMUP: usize = 5;
/// Resolution the CPU kernels are measured at (the worst case for a live 1080p camera).
const KERNEL_FRAME: (usize, usize) = (1920, 1080);
const KERNEL_ITERATIONS: usize = 200;

pub fn run(options: &Options, source: &Path, target: &Path, iterations: usize, profile: bool) -> Result<()> {
    let mut processor = processor(options, source)?;
    let settings = options.settings();
    let image = Frame::open(target)?;
    for _ in 0..WARMUP {
        processor.process(&mut image.clone(), &settings, true)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut frame = image.clone();
    for _ in 0..iterations {
        frame.data.copy_from_slice(&image.data);
        samples.push(processor.process(&mut frame, &settings, true)?);
    }
    let mean = samples.iter().copied().fold(Stats::default(), |mut acc, s| {
        acc += s;
        acc
    }) / iterations.max(1) as f32;
    println!("{}x{} on {}: {mean}", image.width, image.height, processor.backend());
    if profile {
        print_percentiles(&samples);
        print_kernels(&image);
    }
    Ok(())
}

/// A named per-frame timing.
type Stage = (&'static str, fn(&Stats) -> f32);

/// p50 / p95 / max of every stage.
fn print_percentiles(samples: &[Stats]) {
    let stages: [Stage; 4] =
        [("detect", |s| s.detect_ms), ("swap", |s| s.swap_ms), ("enhance", |s| s.enhance_ms), ("total", |s| s.total_ms)];
    println!("{:<10} {:>8} {:>8} {:>8}", "stage", "p50", "p95", "max");
    for (name, stage) in stages {
        let mut values: Vec<f32> = samples.iter().map(stage).collect();
        values.sort_by(f32::total_cmp);
        println!("{name:<10} {:>8.2} {:>8.2} {:>8.2}", percentile(&values, 0.5), percentile(&values, 0.95), percentile(&values, 1.0));
    }
}

/// Value at quantile `q` of an ascending slice (nearest rank).
fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[rank]
}

/// Time the CPU image kernels on a 1080p upscale of `image`, with a face-sized crop.
fn print_kernels(image: &Frame) {
    let frame = upscale(image, KERNEL_FRAME.0, KERNEL_FRAME.1);
    // A 400 px face in the middle of the frame: a typical close-up webcam framing.
    let (cx, cy, half) = (frame.width as f32 / 2.0, frame.height as f32 / 2.0, 200.0);
    let landmarks = Template::Arcface128.points().map(|[x, y]| [cx + (x - 0.5) * 2.0 * half, cy + (y - 0.5) * 2.0 * half]);
    println!("CPU kernels at {}x{} ({KERNEL_ITERATIONS} runs):", frame.width, frame.height);
    for size in [128, 256] {
        let to_crop = Template::Arcface128.align(&landmarks, size);
        let crop = warp_crop(&frame, &to_crop, size);
        let mask = box_mask(size, 0.3, [0.0; 4]);
        kernel(&format!("warp_crop {size}"), || drop(warp_crop(&frame, &to_crop, size)));
        let mut target = frame.clone();
        kernel(&format!("paste_back {size}"), || paste_back(&mut target, &crop, &mask, size, &to_crop, 1.0));
    }
    for size in [160, 320, 640] {
        kernel(&format!("letterbox {size}"), || drop(letterbox_nchw(&frame, size)));
    }
    kernel("to_rgba", || drop(frame.to_rgba()));
    let mut mirrored = frame.clone();
    kernel("mirror", || mirrored.mirror());
}

/// Print the median and best time of `f`; the best is robust against a busy machine.
fn kernel(name: &str, mut f: impl FnMut()) {
    f();
    let mut times: Vec<f32> = (0..KERNEL_ITERATIONS)
        .map(|_| {
            let started = Instant::now();
            f();
            started.elapsed().as_secs_f32() * 1000.0
        })
        .collect();
    times.sort_by(f32::total_cmp);
    println!("  {name:<16} p50 {:>7.3} ms   min {:>7.3} ms", percentile(&times, 0.5), percentile(&times, 0.0));
}

/// Nearest-neighbour resize; good enough to get realistic frame content at a new size.
fn upscale(image: &Frame, width: usize, height: usize) -> Frame {
    let mut out = Frame::new(width, height);
    for (y, row) in out.data.chunks_exact_mut(width * Frame::CHANNELS).enumerate() {
        let sy = y * image.height / height;
        for (x, px) in row.as_chunks_mut::<{ Frame::CHANNELS }>().0.iter_mut().enumerate() {
            let sx = x * image.width / width;
            let at = (sy * image.width + sx) * Frame::CHANNELS;
            px.copy_from_slice(&image.data[at..at + Frame::CHANNELS]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn percentile_picks_nearest_rank() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
