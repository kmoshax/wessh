use super::presenter::{Control, Presenter};
use super::state::{Image, Phase, Shared};
use super::{Command, EngineOptions};
use crossbeam_channel::{Receiver, TryRecvError};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wajh_core::{Engine, Frame, Processor};
use wajh_media::{Camera, FrameSink, LatestFrame};

const THUMBNAIL: u32 = 256;
const FPS_WINDOW: Duration = Duration::from_millis(500);

/// The camera, which exists only while live.
struct Stream {
    _camera: Camera,
}

pub(super) fn run(engine: &EngineOptions, shared: &Arc<Shared>, inbox: &Receiver<Command>, repaint: &Arc<dyn Fn() + Send + Sync>) {
    let mut processor = match load(engine, shared) {
        Ok(p) => p,
        Err(e) => {
            shared.update(|s| s.phase = Phase::Failed(format!("{e:#}")));
            repaint();
            return;
        }
    };
    let presenter = match Presenter::spawn(shared.clone(), repaint.clone()) {
        Ok(p) => p,
        Err(e) => {
            shared.update(|s| s.phase = Phase::Failed(format!("presenter thread: {e}")));
            repaint();
            return;
        }
    };
    shared.update(|s| s.phase = Phase::Ready);
    repaint();

    let mailbox = Arc::new(LatestFrame::default());
    let mut stream: Option<Stream> = None;
    let mut seen = 0;
    let (mut window, mut frames) = (Instant::now(), 0u32);

    loop {
        let command = if stream.is_some() {
            match inbox.try_recv() {
                Ok(c) => Some(c),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => return,
            }
        } else {
            match inbox.recv() {
                Ok(c) => Some(c),
                Err(_) => return,
            }
        };
        if let Some(command) = command {
            handle(command, &mut processor, &mut stream, &mailbox, shared, &presenter);
            repaint();
        }
        if stream.is_none() {
            continue;
        }
        let Some(mut frame) = mailbox.take_newer(&mut seen, Duration::from_millis(100)) else { continue };

        let settings = shared.settings.lock().clone();
        match processor.process(&mut frame, &settings, true) {
            Ok(stats) => {
                frames += 1;
                let elapsed = window.elapsed();
                shared.update(|s| {
                    s.stats = stats;
                    if elapsed >= FPS_WINDOW {
                        s.fps = frames as f32 / elapsed.as_secs_f32();
                    }
                });
                if elapsed >= FPS_WINDOW {
                    (window, frames) = (Instant::now(), 0);
                }
            }
            Err(e) => shared.update(|s| s.notice = Some(format!("{e:#}"))),
        }
        presenter.present(frame);
    }
}

fn load(engine: &EngineOptions, shared: &Shared) -> anyhow::Result<Processor> {
    let engine = Engine::init(engine.runtime.as_deref(), engine.device)?;
    shared.update(|s| s.backend = engine.backend().to_string());
    let mut processor = Processor::new(engine)?;
    processor.prepare(&shared.settings.lock().clone())?;
    Ok(processor)
}

fn handle(
    command: Command,
    processor: &mut Processor,
    stream: &mut Option<Stream>,
    mailbox: &Arc<LatestFrame>,
    shared: &Shared,
    presenter: &Presenter,
) {
    match command {
        Command::SetSource(path) => match load_source(processor, &path) {
            Ok(thumbnail) => {
                *shared.source_thumbnail.lock() = Some(thumbnail);
                shared.update(|s| {
                    s.has_source = true;
                    s.source_name = path.file_name().map(|n| n.to_string_lossy().into_owned());
                    s.source_path = Some(path.clone());
                    s.notice = None;
                });
            }
            Err(e) => shared.update(|s| s.notice = Some(format!("{e:#}"))),
        },
        Command::Start { camera, output } => {
            *stream = None; // release the device before reopening it
            presenter.control(Control::Stream { output: None, camera: None });
            match Camera::start(&camera, mailbox.clone()) {
                Ok(capture) => {
                    let output = output.and_then(|target| match FrameSink::live(&target, camera.width, camera.height, camera.fps) {
                        Ok(sink) => Some(sink),
                        Err(e) => {
                            shared.update(|s| s.notice = Some(format!("virtual camera: {e}")));
                            None
                        }
                    });
                    let output_active = output.is_some();
                    presenter.control(Control::Stream { output, camera: Some(mailbox.clone()) });
                    *stream = Some(Stream { _camera: capture });
                    shared.update(|s| {
                        s.phase = Phase::Live;
                        s.output_active = output_active;
                    });
                }
                Err(e) => shared.update(|s| s.notice = Some(format!("camera: {e}"))),
            }
        }
        Command::Stop => {
            *stream = None;
            presenter.control(Control::Stream { output: None, camera: None });
            shared.update(|s| {
                s.phase = Phase::Ready;
                s.output_active = false;
                s.fps = 0.0;
            });
        }
    }
}

fn load_source(processor: &mut Processor, path: &Path) -> anyhow::Result<Image> {
    let rgb = image::open(path)?.to_rgb8();
    processor.set_source(&Frame::from_rgb(&rgb))?;
    let (w, h) = rgb.dimensions();
    let (tw, th) = if w >= h { (THUMBNAIL, (THUMBNAIL * h / w).max(1)) } else { ((THUMBNAIL * w / h).max(1), THUMBNAIL) };
    let thumbnail = image::DynamicImage::ImageRgb8(image::imageops::thumbnail(&rgb, tw, th)).to_rgba8();
    Ok(Image { width: thumbnail.width() as usize, height: thumbnail.height() as usize, rgba: thumbnail.into_raw() })
}
