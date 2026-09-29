mod engine;
mod face;
mod geom;
mod gui;
mod imgops;
mod io;
mod models;
mod onnx_proto;
mod processor;
mod smoothing;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use engine::{Device, Engine};
use imgops::Frame;
use models::SwapModel;
use processor::{Processor, Settings};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "wajh", version, about = "Real-time face swap. Run without a subcommand to open the app.")]
struct Cli {
    #[command(flatten)]
    common: Common,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Args, Clone)]
struct Common {
    /// Path to the ONNX Runtime library (libonnxruntime.so / onnxruntime.dll).
    #[arg(long, global = true)]
    ort: Option<PathBuf>,
    /// Inference backend.
    #[arg(long, global = true, value_enum, default_value_t = Device::Auto)]
    device: Device,
    /// Face swap model.
    #[arg(long, global = true, value_enum, default_value_t = SwapModel::Inswapper)]
    model: SwapModel,
    /// Run the GPEN face enhancer after swapping.
    #[arg(long, global = true)]
    enhance: bool,
    /// Swap every face instead of only the largest one.
    #[arg(long, global = true)]
    many_faces: bool,
    /// Detector input size (smaller = faster, 320 is plenty for close-up webcams).
    #[arg(long, global = true, default_value_t = 640)]
    detector_size: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Swap faces in a single image.
    Image {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Swap faces in a video file (audio is kept).
    Video {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: String,
        #[arg(short, long)]
        output: String,
        /// ffmpeg video encoder (default: NVENC/VideoToolbox when available, else libx264).
        #[arg(long)]
        encoder: Option<String>,
    },
    /// Headless live mode: webcam -> swap -> virtual camera / stream.
    Cam {
        #[arg(short, long)]
        source: PathBuf,
        #[command(flatten)]
        cam: CamArgs,
        /// Output: /dev/videoN (v4l2loopback) or udp://127.0.0.1:27000 (OBS media source).
        #[arg(long)]
        out: Option<String>,
    },
    /// Measure per-stage latency on an image.
    Bench {
        #[arg(short, long)]
        source: PathBuf,
        #[arg(short, long)]
        target: PathBuf,
        #[arg(short = 'n', long, default_value_t = 100)]
        iterations: usize,
    },
    /// Download the models used by the chosen options.
    Models,
}

#[derive(Args, Clone)]
struct CamArgs {
    /// Camera device (Linux: /dev/video0, macOS: index, Windows: video=<name>).
    #[arg(long, default_value_t = io::default_camera_device())]
    camera: String,
    #[arg(long, default_value_t = 1280)]
    width: usize,
    #[arg(long, default_value_t = 720)]
    height: usize,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// Mirror the image like a selfie camera.
    #[arg(long)]
    mirror: bool,
}

impl Common {
    fn settings(&self) -> Settings {
        Settings {
            swap_model: self.model,
            enhance: self.enhance,
            many_faces: self.many_faces,
            detector_size: self.detector_size,
            ..Settings::default()
        }
    }

    fn processor(&self, source: Option<&PathBuf>) -> Result<Processor> {
        let engine = Engine::init(self.ort.as_deref(), self.device)?;
        eprintln!("backend: {}", engine.backend);
        let mut p = Processor::new(engine)?;
        p.prepare(&self.settings())?;
        if let Some(src) = source {
            p.set_source(&load_image(src)?)?;
        }
        Ok(p)
    }
}

fn load_image(path: &PathBuf) -> Result<Frame> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
    Ok(Frame::from_rgb_image(&img))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let c = cli.common;
    match cli.command {
        None => gui::run(c.ort.clone(), c.device, c.settings()),
        Some(Cmd::Models) => {
            for m in [&models::DETECTOR, &models::RECOGNIZER, &c.model.file()] {
                println!("{}", models::ensure(m)?.display());
            }
            if c.enhance {
                println!("{}", models::ensure(&models::ENHANCER)?.display());
            }
            Ok(())
        }
        Some(Cmd::Image { source, target, output }) => {
            let mut p = c.processor(Some(&source))?;
            let mut frame = load_image(&target)?;
            let stats = p.process(&mut frame, &c.settings(), false)?;
            frame.to_rgb_image().save(&output)?;
            eprintln!("{} face(s) swapped in {:.1} ms -> {}", stats.faces, stats.total_ms, output.display());
            Ok(())
        }
        Some(Cmd::Bench { source, target, iterations }) => {
            let mut p = c.processor(Some(&source))?;
            let base = load_image(&target)?;
            let settings = c.settings();
            let mut frame = base.clone();
            p.process(&mut frame, &settings, true)?; // warm-up (TensorRT builds engines here)
            let mut acc = processor::Stats::default();
            for _ in 0..iterations {
                let mut frame = base.clone();
                let s = p.process(&mut frame, &settings, true)?;
                acc.detect_ms += s.detect_ms;
                acc.swap_ms += s.swap_ms;
                acc.enhance_ms += s.enhance_ms;
                acc.total_ms += s.total_ms;
            }
            let n = iterations as f32;
            println!(
                "{}x{} on {}: detect {:.2} ms | swap+paste {:.2} ms | enhance {:.2} ms | total {:.2} ms ({:.1} fps)",
                base.width,
                base.height,
                p.backend(),
                acc.detect_ms / n,
                acc.swap_ms / n,
                acc.enhance_ms / n,
                acc.total_ms / n,
                1000.0 * n / acc.total_ms
            );
            Ok(())
        }
        Some(Cmd::Video { source, target, output, encoder }) => run_video(&c, &source, &target, &output, encoder),
        Some(Cmd::Cam { source, cam, out }) => run_cam(&c, &source, &cam, out),
    }
}

/// Decode, swap and encode run on separate threads so ffmpeg never waits on inference.
fn run_video(c: &Common, source: &PathBuf, target: &str, output: &str, encoder: Option<String>) -> Result<()> {
    let info = io::probe(target)?;
    let encoder = encoder.unwrap_or_else(|| io::best_encoder().to_string());
    eprintln!("{}x{} @ {:.2} fps, encoder {encoder}", info.width, info.height, info.fps);
    let mut p = c.processor(Some(source))?;
    let settings = c.settings();

    let (dec_tx, dec_rx) = crossbeam_channel::bounded::<Frame>(8);
    let (enc_tx, enc_rx) = crossbeam_channel::bounded::<Frame>(8);
    let mut reader = io::VideoReader::open(target, &info)?;
    let decoder = std::thread::spawn(move || {
        while let Some(f) = reader.next() {
            if dec_tx.send(f).is_err() {
                break;
            }
        }
    });
    let writer = io::video_writer(output, target, &info, &encoder)?;
    let encoder_thread = std::thread::spawn(move || -> Result<()> {
        let mut writer = writer;
        for f in enc_rx {
            writer.write(&f)?;
        }
        writer.finish()
    });

    let start = Instant::now();
    let mut n = 0u64;
    for mut frame in dec_rx {
        p.process(&mut frame, &settings, false)?;
        n += 1;
        if n % 30 == 0 {
            let fps = n as f64 / start.elapsed().as_secs_f64();
            match info.frames {
                Some(total) => eprint!("\r{n}/{total} frames, {fps:.1} fps   "),
                None => eprint!("\r{n} frames, {fps:.1} fps   "),
            }
        }
        if enc_tx.send(frame).is_err() {
            break;
        }
    }
    drop(enc_tx);
    let _ = decoder.join();
    encoder_thread.join().expect("encoder thread")?;
    eprintln!("\ndone: {n} frames in {:.1}s -> {output}", start.elapsed().as_secs_f64());
    Ok(())
}

fn run_cam(c: &Common, source: &PathBuf, cam: &CamArgs, out: Option<String>) -> Result<()> {
    let mut p = c.processor(Some(source))?;
    let settings = Settings { mirror: cam.mirror, ..c.settings() };
    let latest = Arc::new(io::LatestFrame::default());
    let _camera = io::Camera::start(&io::CameraConfig { device: cam.camera.clone(), width: cam.width, height: cam.height, fps: cam.fps }, latest.clone())?;
    let mut sink = match &out {
        Some(t) => Some(io::FrameSink::virtual_camera(t, cam.width, cam.height, cam.fps)?),
        None => None,
    };
    eprintln!("running, Ctrl+C to stop");
    let (mut seq, mut frames, mut window) = (0u64, 0u32, Instant::now());
    let mut acc = processor::Stats::default();
    loop {
        let Some(mut frame) = latest.take_newer(&mut seq, Duration::from_secs(2)) else {
            anyhow::bail!("camera stopped delivering frames");
        };
        let s = p.process(&mut frame, &settings, true)?;
        acc.total_ms += s.total_ms;
        acc.detect_ms += s.detect_ms;
        acc.swap_ms += s.swap_ms;
        acc.enhance_ms += s.enhance_ms;
        if let Some(sink) = sink.as_mut() {
            sink.write(&frame)?;
        }
        frames += 1;
        if window.elapsed() >= Duration::from_secs(1) {
            let n = frames as f32;
            eprint!(
                "\r{:.1} fps | detect {:.1} ms | swap {:.1} ms | enhance {:.1} ms | total {:.1} ms   ",
                n / window.elapsed().as_secs_f32(),
                acc.detect_ms / n,
                acc.swap_ms / n,
                acc.enhance_ms / n,
                acc.total_ms / n
            );
            (frames, window, acc) = (0, Instant::now(), processor::Stats::default());
        }
    }
}
