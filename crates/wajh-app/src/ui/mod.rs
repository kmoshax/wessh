//! Desktop UI (egui). Talks to the worker only through [`LiveSession`].

use crate::cli::Options;
use crate::live::{EngineOptions, Image, LiveSession, Phase};
use eframe::egui;
use std::path::PathBuf;
use wajh_core::assets::SwapModel;
use wajh_media::{CameraConfig, OutputTarget};

pub fn run(options: Options) -> anyhow::Result<()> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1320.0, 820.0]).with_title("wajh"),
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(
        "wajh",
        native,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            let session = LiveSession::spawn(
                EngineOptions { runtime: options.ort.clone(), device: options.device },
                options.settings(),
                move || ctx.request_repaint(),
            );
            Ok(Box::new(App::new(session)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

struct App {
    session: LiveSession,
    preview: Option<egui::TextureHandle>,
    source: Option<egui::TextureHandle>,
    source_path: String,
    camera: CameraConfig,
    output_enabled: bool,
    output_target: String,
}

impl App {
    fn new(session: LiveSession) -> Self {
        Self {
            session,
            preview: None,
            source: None,
            source_path: String::new(),
            camera: CameraConfig::default(),
            output_enabled: false,
            output_target: OutputTarget::default_for_platform().as_str().to_owned(),
        }
    }
}

fn upload(ctx: &egui::Context, slot: &mut Option<egui::TextureHandle>, name: &str, image: &Image) {
    let pixels = egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba);
    match slot {
        Some(texture) => texture.set(pixels, egui::TextureOptions::LINEAR),
        None => *slot = Some(ctx.load_texture(name, pixels, egui::TextureOptions::LINEAR)),
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.source_path = path.display().to_string();
            self.session.set_source(path);
        }
        if let Some(image) = self.session.take_preview() {
            upload(ctx, &mut self.preview, "preview", &image);
        }
        if let Some(image) = self.session.take_source_thumbnail() {
            upload(ctx, &mut self.source, "source", &image);
        }
        let snapshot = self.session.snapshot();
        let live = snapshot.phase == Phase::Live;
        let mut settings = self.session.settings();

        egui::SidePanel::left("controls").exact_width(300.0).show(ctx, |ui| {
            ui.heading("wajh");
            ui.label(format!("backend: {}", snapshot.backend));
            if let Some(texture) = &self.source {
                ui.add(egui::Image::new(texture).max_height(180.0));
            }
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.source_path);
                if ui.button("Load").clicked() {
                    self.session.set_source(PathBuf::from(self.source_path.trim()));
                }
            });
            ui.add_enabled_ui(!live, |ui| {
                ui.text_edit_singleline(&mut self.camera.device);
                ui.checkbox(&mut self.output_enabled, "Virtual camera");
                if self.output_enabled {
                    ui.text_edit_singleline(&mut self.output_target);
                }
            });
            if ui.button(if live { "Stop" } else { "Start" }).clicked() {
                if live {
                    self.session.stop();
                } else {
                    let output = self.output_enabled.then(|| OutputTarget::parse(&self.output_target));
                    self.session.start(self.camera.clone(), output);
                }
            }
            egui::ComboBox::from_label("Model").selected_text(settings.swap_model.label()).show_ui(ui, |ui| {
                for model in SwapModel::ALL {
                    ui.selectable_value(&mut settings.swap_model, model, model.label());
                }
            });
            ui.checkbox(&mut settings.enhance, "Enhance");
            ui.checkbox(&mut settings.mirror, "Mirror");
            ui.checkbox(&mut settings.many_faces, "All faces");
            if live {
                ui.monospace(format!("{:.1} fps  {}", snapshot.fps, snapshot.stats));
            }
            if let Phase::Failed(error) = &snapshot.phase {
                ui.colored_label(egui::Color32::RED, error);
            }
            if let Some(notice) = &snapshot.notice {
                ui.label(notice);
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.preview {
                let scale = (ui.available_size() / texture.size_vec2()).min_elem();
                ui.centered_and_justified(|ui| ui.add(egui::Image::new(texture).fit_to_exact_size(texture.size_vec2() * scale)));
            }
        });
        if settings != self.session.settings() {
            self.session.set_settings(settings);
        }
    }
}
