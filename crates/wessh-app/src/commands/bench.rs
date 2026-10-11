//! `wessh bench`: end-to-end per-stage latency, plus an optional CPU-kernel breakdown.

use super::processor;
use crate::cli::Options;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wessh_core::geometry::Template;
use wessh_core::image::{box_mask, letterbox_nchw, paste_back, warp_crop};
use wessh_core::{Frame, Stats};

/// Iterations that pay for cuDNN autotuning, TensorRT engine builds and CUDA graph capture.
const WARMUP: usize = 5;
/// Resolution the CPU kernels are measured at (the worst case for a live 1080p camera).
const KERNEL_FRAME: (usize, usize) = (1920, 1080);
const KERNEL_ITERATIONS: usize = 200;

pub fn run(options: &Options, source: &[PathBuf], target: &Path, iterations: usize, profile: bool, motion: bool) -> Result<()> {
    let mut processor = processor(options, source)?;
    let settings = options.settings();
    let image = Frame::open(target)?;
    let mut frame = image.clone();
    for i in 0..WARMUP {
        load_frame(&image, &mut frame, motion.then_some(i));
        processor.process(&mut frame, &settings, true)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for i in 0..iterations {
        load_frame(&image, &mut frame, motion.then_some(WARMUP + i));
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

/// Camera path of `--motion`: a slow pan (amplitude in pixels, frequency in Hz) and zoom, at
/// an assumed 30 fps. Large enough that faces move tens of pixels per second, small enough
/// that the tracker keeps them.
const MOTION_FPS: f32 = 30.0;
const PAN: [(f32, f32); 2] = [(40.0, 0.5), (20.0, 0.3)];
const ZOOM: (f32, f32) = (0.06, 0.2);

/// Fill `frame` with `image`, or with frame `index` of the `--motion` camera path over it.
fn load_frame(image: &Frame, frame: &mut Frame, index: Option<usize>) {
    let Some(index) = index else {
        frame.data.copy_from_slice(&image.data);
        return;
    };
    let t = index as f32 / MOTION_FPS;
    let wave = |(amplitude, hz): (f32, f32)| amplitude * (std::f32::consts::TAU * hz * t).sin();
    let (dx, dy, zoom) = (wave(PAN[0]), wave(PAN[1]), 1.0 + ZOOM.0 + wave(ZOOM));
    let (cx, cy) = (image.width as f32 / 2.0, image.height as f32 / 2.0);
    for (y, row) in frame.data.chunks_exact_mut(frame.width * Frame::CHANNELS).enumerate() {
        let sy = ((y as f32 - cy) / zoom + cy + dy).clamp(0.0, (image.height - 1) as f32);
        for (x, px) in row.as_chunks_mut::<{ Frame::CHANNELS }>().0.iter_mut().enumerate() {
            let sx = ((x as f32 - cx) / zoom + cx + dx).clamp(0.0, (image.width - 1) as f32);
            // Bilinear, so sub-pixel motion is smooth.
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(image.width - 1), (y0 + 1).min(image.height - 1));
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let at = |x: usize, y: usize, c: usize| f32::from(image.data[(y * image.width + x) * Frame::CHANNELS + c]);
            for (c, v) in px.iter_mut().enumerate() {
                let top = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * fx;
                let bottom = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * fx;
                *v = (top + (bottom - top) * fy).round() as u8;
            }
        }
    }
}

/// A named per-frame timing.
type Stage = (&'static str, fn(&Stats) -> f32);

/// p50 / p95 / max of every stage, and how often the live-mode mask cache served a mask.
fn print_percentiles(samples: &[Stats]) {
    let stages: [Stage; 7] = [
        ("detect", |s| s.detect_ms),
        ("swap", |s| s.swap_ms),
        ("masks", |s| s.mask_ms),
        ("occlusion", |s| s.occlusion_ms),
        ("parser", |s| s.parser_ms),
        ("enhance", |s| s.enhance_ms),
        ("total", |s| s.total_ms),
    ];
    println!("{:<10} {:>8} {:>8} {:>8}", "stage", "p50", "p95", "max");
    for (name, stage) in stages {
        let mut values: Vec<f32> = samples.iter().map(stage).collect();
        values.sort_by(f32::total_cmp);
        println!("{name:<10} {:>8.2} {:>8.2} {:>8.2}", percentile(&values, 0.5), percentile(&values, 0.95), percentile(&values, 1.0));
    }
    let (computed, reused) = samples.iter().fold((0, 0), |(c, r), s| (c + s.masks_computed, r + s.masks_reused));
    if computed + reused > 0 {
        println!(
            "mask cache: {reused} of {} model masks reused ({:.0}%)",
            computed + reused,
            100.0 * reused as f32 / (computed + reused) as f32
        );
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
