//! Desktop app: egui front-end, with capture/inference/output on a dedicated worker thread.

use crate::engine::{Device, Engine};
use crate::imgops::{self, Frame};
use crate::io::{self, Camera, CameraConfig, FrameSink, LatestFrame};
use crate::models::SwapModel;
use crate::processor::{Processor, Settings, Stats};
use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

enum Command {
    SetSource(PathBuf),
    Start { camera: CameraConfig, vcam: Option<String> },
    Stop,
}

struct Preview {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

#[derive(Default)]
struct Shared {
    settings: Mutex<Settings>,
    preview: Mutex<Option<Preview>>,
    stats: Mutex<(Stats, f32)>,
    status: Mutex<String>,
    backend: Mutex<String>,
    running: Mutex<bool>,
    source_preview: Mutex<Option<Preview>>,
}

fn set_status(shared: &Shared, msg: impl Into<String>) {
    *shared.status.lock() = msg.into();
}

fn worker(shared: Arc<Shared>, rx: Receiver<Command>, ctx: egui::Context, ort: Option<PathBuf>, device: Device) {
    set_status(&shared, "loading models…");
    ctx.request_repaint();
    let mut processor = match Engine::init(ort.as_deref(), device).and_then(|e| {
        *shared.backend.lock() = e.backend.to_string();
        let mut p = Processor::new(e)?;
        p.prepare(&shared.settings.lock().clone())?;
        Ok(p)
    }) {
        Ok(p) => p,
        Err(e) => {
            set_status(&shared, format!("error: {e:#}"));
            ctx.request_repaint();
            return;
        }
    };
    set_status(&shared, "ready: drop a source face image, then press Start");
    ctx.request_repaint();

    let latest = Arc::new(LatestFrame::default());
    let mut camera: Option<Camera> = None;
    let mut sink: Option<FrameSink> = None;
    let mut seq = 0u64;
    let (mut frames, mut window) = (0u32, Instant::now());

    loop {
        let cmd = if camera.is_some() {
            match rx.try_recv() {
                Ok(c) => Some(c),
                Err(crossbeam_channel::TryRecvError::Empty) => None,
                Err(_) => return,
            }
        } else {
            match rx.recv() {
                Ok(c) => Some(c),
                Err(_) => return,
            }
        };
        match cmd {
            Some(Command::SetSource(path)) => {
                let res = image::open(&path).map_err(anyhow::Error::from).and_then(|img| {
                    let img = img.to_rgb8();
                    let frame = Frame::from_rgb_image(&img);
                    processor.set_source(&frame)?;
                    let thumb = image::imageops::thumbnail(&img, 256, (256 * img.height() / img.width().max(1)).max(1));
                    let rgba = image::DynamicImage::ImageRgb8(thumb).to_rgba8();
                    *shared.source_preview.lock() = Some(Preview { width: rgba.width() as usize, height: rgba.height() as usize, rgba: rgba.into_raw() });
                    Ok(())
                });
                match res {
                    Ok(()) => set_status(&shared, format!("source: {}", path.display())),
                    Err(e) => set_status(&shared, format!("source error: {e:#}")),
                }
                ctx.request_repaint();
            }
            Some(Command::Start { camera: cfg, vcam }) => {
                camera = None;
                sink = None;
                match Camera::start(&cfg, latest.clone()) {
                    Ok(c) => camera = Some(c),
                    Err(e) => set_status(&shared, format!("camera error: {e:#}")),
                }
                if let (Some(target), Some(_)) = (vcam, camera.as_ref()) {
                    match FrameSink::virtual_camera(&target, cfg.width, cfg.height, cfg.fps) {
                        Ok(s) => sink = Some(s),
                        Err(e) => set_status(&shared, format!("virtual camera error: {e:#}")),
                    }
                }
                *shared.running.lock() = camera.is_some();
                if camera.is_some() {
                    set_status(&shared, "live");
                }
                ctx.request_repaint();
            }
            Some(Command::Stop) => {
                camera = None;
                sink = None;
                *shared.running.lock() = false;
                set_status(&shared, "stopped");
                ctx.request_repaint();
            }
            None => {}
        }
        if camera.is_none() {
            continue;
        }
        let Some(mut frame) = latest.take_newer(&mut seq, Duration::from_millis(100)) else {
            continue;
        };
        let settings = shared.settings.lock().clone();
        match processor.process(&mut frame, &settings, true) {
            Ok(stats) => {
                frames += 1;
                let elapsed = window.elapsed().as_secs_f32();
                let mut st = shared.stats.lock();
                st.0 = stats;
                if elapsed >= 0.5 {
                    st.1 = frames as f32 / elapsed;
                    (frames, window) = (0, Instant::now());
                }
            }
            Err(e) => set_status(&shared, format!("error: {e:#}")),
        }
        if let Some(s) = sink.as_mut() {
            if let Err(e) = s.write(&frame) {
                set_status(&shared, format!("virtual camera error: {e:#}"));
                sink = None;
            }
        }
        *shared.preview.lock() = Some(Preview { width: frame.width, height: frame.height, rgba: imgops::to_rgba(&frame) });
        ctx.request_repaint();
    }
}

struct App {
    shared: Arc<Shared>,
    tx: Sender<Command>,
    texture: Option<egui::TextureHandle>,
    source_texture: Option<egui::TextureHandle>,
    source_path: String,
    camera: String,
    resolution: (usize, usize),
    fps: u32,
    vcam_enabled: bool,
    vcam_target: String,
}

impl App {
    fn start(&self) {
        let camera = CameraConfig { device: self.camera.clone(), width: self.resolution.0, height: self.resolution.1, fps: self.fps };
        let vcam = self.vcam_enabled.then(|| self.vcam_target.clone());
        let _ = self.tx.send(Command::Start { camera, vcam });
    }
}

const RESOLUTIONS: [(usize, usize); 3] = [(640, 480), (1280, 720), (1920, 1080)];

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drag & drop a source image anywhere on the window.
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.source_path = path.display().to_string();
            let _ = self.tx.send(Command::SetSource(path));
        }
        if let Some(p) = self.shared.preview.lock().take() {
            let img = egui::ColorImage::from_rgba_unmultiplied([p.width, p.height], &p.rgba);
            match &mut self.texture {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.texture = Some(ctx.load_texture("preview", img, egui::TextureOptions::LINEAR)),
            }
        }
        if let Some(p) = self.shared.source_preview.lock().take() {
            let img = egui::ColorImage::from_rgba_unmultiplied([p.width, p.height], &p.rgba);
            self.source_texture = Some(ctx.load_texture("source", img, egui::TextureOptions::LINEAR));
        }

        let running = *self.shared.running.lock();
        let mut settings = self.shared.settings.lock().clone();
        let before = format!("{settings:?}");

        egui::SidePanel::left("controls").resizable(false).exact_width(290.0).show(ctx, |ui| {
            ui.heading("wajh");
            ui.label(egui::RichText::new(format!("backend: {}", self.shared.backend.lock())).weak());
            ui.separator();

            ui.strong("Source face");
            if let Some(t) = &self.source_texture {
                ui.add(egui::Image::new(t).max_width(270.0).max_height(200.0));
            } else {
                ui.label("Drop an image on the window, or enter a path:");
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.source_path).desired_width(200.0));
                if ui.button("Load").clicked() && !self.source_path.is_empty() {
                    let _ = self.tx.send(Command::SetSource(PathBuf::from(self.source_path.trim())));
                }
            });
            ui.separator();

            ui.strong("Camera");
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Device");
                    ui.text_edit_singleline(&mut self.camera);
                });
                egui::ComboBox::from_label("Resolution")
                    .selected_text(format!("{}x{}", self.resolution.0, self.resolution.1))
                    .show_ui(ui, |ui| {
                        for r in RESOLUTIONS {
                            ui.selectable_value(&mut self.resolution, r, format!("{}x{}", r.0, r.1));
                        }
                    });
                ui.add(egui::Slider::new(&mut self.fps, 15..=60).text("FPS"));
                ui.checkbox(&mut self.vcam_enabled, "Virtual camera output");
                if self.vcam_enabled {
                    ui.text_edit_singleline(&mut self.vcam_target);
                }
            });
            let label = if running { "⏹ Stop" } else { "▶ Start" };
            if ui.add_sized([270.0, 32.0], egui::Button::new(label)).clicked() {
                if running {
                    let _ = self.tx.send(Command::Stop);
                } else {
                    self.start();
                }
            }
            ui.separator();

            ui.strong("Swap");
            egui::ComboBox::from_label("Model").selected_text(settings.swap_model.label()).show_ui(ui, |ui| {
                for m in SwapModel::ALL {
                    ui.selectable_value(&mut settings.swap_model, m, m.label());
                }
            });
            ui.checkbox(&mut settings.many_faces, "Swap all faces");
            ui.checkbox(&mut settings.mirror, "Mirror");
            ui.checkbox(&mut settings.smooth, "Stabilise (One Euro)");
            ui.add(egui::Slider::new(&mut settings.mask_blur, 0.0..=1.0).text("Mask blur"));
            ui.add(egui::Slider::new(&mut settings.mask_padding[2], 0.0..=40.0).text("Mask bottom pad %"));
            egui::ComboBox::from_label("Detector size").selected_text(settings.detector_size.to_string()).show_ui(ui, |ui| {
                for s in [320, 480, 640] {
                    ui.selectable_value(&mut settings.detector_size, s, s.to_string());
                }
            });
            ui.checkbox(&mut settings.enhance, "Enhance (GPEN 256)");
            if settings.enhance {
                ui.add(egui::Slider::new(&mut settings.enhance_blend, 0.0..=1.0).text("Enhance blend"));
            }
            ui.separator();

            let (stats, fps) = *self.shared.stats.lock();
            if running {
                ui.monospace(format!("{fps:5.1} fps   {} face(s)", stats.faces));
                ui.monospace(format!("detect  {:6.2} ms", stats.detect_ms));
                ui.monospace(format!("swap    {:6.2} ms", stats.swap_ms));
                if settings.enhance {
                    ui.monospace(format!("enhance {:6.2} ms", stats.enhance_ms));
                }
                ui.monospace(format!("total   {:6.2} ms", stats.total_ms));
            }
            ui.add_space(8.0);
            ui.label(egui::RichText::new(self.shared.status.lock().clone()).small());
        });

        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(egui::Color32::from_gray(16))).show(ctx, |ui| {
            if let Some(t) = &self.texture {
                let avail = ui.available_size();
                let size = t.size_vec2();
                let scale = (avail.x / size.x).min(avail.y / size.y);
                ui.centered_and_justified(|ui| ui.add(egui::Image::new(t).fit_to_exact_size(size * scale)));
            } else {
                ui.centered_and_justified(|ui| ui.label("preview"));
            }
        });

        if format!("{settings:?}") != before {
            *self.shared.settings.lock() = settings;
        }
    }
}

pub fn run(ort: Option<PathBuf>, device: Device, settings: Settings) -> Result<()> {
    let shared = Arc::new(Shared::default());
    *shared.settings.lock() = settings;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1320.0, 820.0]).with_title("wajh"),
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(
        "wajh",
        options,
        Box::new(move |cc| {
            let (tx, rx) = crossbeam_channel::unbounded();
            let ctx = cc.egui_ctx.clone();
            let s = shared.clone();
            std::thread::Builder::new().name("worker".into()).spawn(move || worker(s, rx, ctx, ort, device))?;
            Ok(Box::new(App {
                shared,
                tx,
                texture: None,
                source_texture: None,
                source_path: String::new(),
                camera: io::default_camera_device(),
                resolution: (1280, 720),
                fps: 30,
                vcam_enabled: false,
                vcam_target: if cfg!(target_os = "linux") { "/dev/video10".into() } else { "udp://127.0.0.1:27000".into() },
            }))
        }),
    )
    .map_err(|e| anyhow!("{e}"))
}
