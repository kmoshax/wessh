//! Desktop UI (egui), styled after Material Design 3. Talks to the worker only through
//! [`LiveSession`].
//!
//! - [`theme`]: HCT colour scheme, type scale, tokens and the egui style mapping.
//! - [`components`]: reusable MD3 widgets.
//! - [`screens`]: the Live / Image / Video destinations.

mod capture;
mod components;
mod file_picker;
mod icons;
mod screens;
mod theme;

use crate::cli::Options;
use crate::live::{EngineOptions, LiveSession, Phase, Snapshot};
use components::navigation::{RAIL_WIDTH, TOP_APP_BAR_HEIGHT};
use components::{Chip, IconButton, drop_overlay, navigation_rail, top_app_bar};
use eframe::egui::{self, CentralPanel, Context, Key, Margin, SidePanel, TopBottomPanel};
use screens::{LiveScreen, Screen, offline_screen};
use theme::{Brightness, ThemeConfig, spacing};

const INITIAL_WINDOW: [f32; 2] = [1320.0, 840.0];
const MIN_WINDOW: [f32; 2] = [560.0, 480.0];
/// Content width from which the settings sheet docks beside the preview
/// (MD3 "expanded" window class starts at 840 dp).
const EXPANDED_BREAKPOINT: f32 = 840.0;
const APP_NAME: &str = "wajh";

/// Open the desktop app.
pub fn run(options: Options) -> anyhow::Result<()> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(INITIAL_WINDOW)
            .with_min_inner_size(MIN_WINDOW)
            .with_title(APP_NAME)
            .with_app_id(APP_NAME),
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(
        APP_NAME,
        native,
        Box::new(move |cc| {
            let engine = EngineOptions { runtime: options.ort.clone(), device: options.device };
            let ctx = cc.egui_ctx.clone();
            let session = LiveSession::spawn(engine, options.settings(), move || ctx.request_repaint());
            Ok(Box::new(App::new(&cc.egui_ctx, session)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

struct App {
    session: LiveSession,
    theme: ThemeConfig,
    screen: Screen,
    live: LiveScreen,
    capture: Option<capture::Capture>,
}

impl App {
    fn new(ctx: &Context, session: LiveSession) -> Self {
        let dev = capture::DevOptions::from_env();
        let brightness = if dev.light { Brightness::Light } else { Brightness::Dark };
        let theme = ThemeConfig { brightness, ..ThemeConfig::default() };
        theme::install_fonts(ctx);
        theme::apply(ctx, theme);
        let mut live = LiveScreen::default();
        if let Some(source) = dev.source {
            live.request_source(ctx, &session, &Snapshot::default(), source);
        }
        Self { session, theme, screen: dev.screen, live, capture: dev.capture }
    }

    fn toggle_brightness(&mut self, ctx: &Context) {
        self.theme.brightness = self.theme.brightness.toggled();
        theme::apply(ctx, self.theme);
    }

    /// Dropped image files become the source face.
    fn handle_drops(&mut self, ctx: &Context, snapshot: &Snapshot) {
        let dropped = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.live.request_source(ctx, &self.session, snapshot, path);
            self.screen = Screen::Live;
        }
    }

    /// Space starts/stops the camera unless a widget has keyboard focus.
    fn handle_shortcuts(&self, ctx: &Context, snapshot: &Snapshot) {
        let space = ctx.input(|i| i.key_pressed(Key::Space));
        if space && ctx.memory(|m| m.focused().is_none()) && self.screen == Screen::Live {
            self.live.toggle_live(&self.session, snapshot);
        }
    }

    fn app_bar(&mut self, ctx: &Context, snapshot: &Snapshot, expanded: bool) {
        let colors = theme::colors(ctx);
        let frame = egui::Frame::new().fill(colors.surface);
        TopBottomPanel::top("app-bar").exact_height(TOP_APP_BAR_HEIGHT).frame(frame).show_separator_line(false).show(ctx, |ui| {
            let mut toggle_theme = false;
            let mut toggle_sheet = false;
            top_app_bar(ui, icons::FACE, APP_NAME, |ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                let (icon, tip) = if colors.is_dark() { (icons::LIGHT_MODE, "Light theme") } else { (icons::DARK_MODE, "Dark theme") };
                toggle_theme = ui.add(IconButton::new(icon, tip)).clicked();
                if !expanded && self.screen == Screen::Live {
                    toggle_sheet = ui.add(IconButton::new(icons::TUNE, "Settings")).clicked();
                }
                backend_chip(ui, snapshot);
            });
            if toggle_theme {
                self.toggle_brightness(ctx);
            }
            if toggle_sheet {
                self.live.toggle_sheet();
            }
        });
    }

    fn rail(&mut self, ctx: &Context) {
        let colors = theme::colors(ctx);
        let frame = egui::Frame::new().fill(colors.surface);
        SidePanel::left("rail").exact_width(RAIL_WIDTH).resizable(false).frame(frame).show_separator_line(false).show(ctx, |ui| {
            navigation_rail(ui, &mut self.screen, &Screen::DESTINATIONS);
        });
    }
}

/// Inference backend readout; tonal while loading, primary once ready.
fn backend_chip(ui: &mut egui::Ui, snapshot: &Snapshot) {
    let (label, icon) = match &snapshot.phase {
        Phase::Loading => ("Loading…".to_owned(), icons::HOURGLASS_TOP),
        Phase::Failed(_) => ("Unavailable".to_owned(), icons::WARNING),
        Phase::Ready | Phase::Live => (snapshot.backend.clone(), icons::MEMORY),
    };
    if !label.is_empty() {
        ui.add(Chip::status(&label, false).icon(icon)).on_hover_text("Inference backend");
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let snapshot = self.session.snapshot();
        self.handle_drops(ctx, &snapshot);
        self.live.sync(ctx, &self.session, &snapshot);
        self.handle_shortcuts(ctx, &snapshot);

        let expanded = ctx.content_rect().width() - RAIL_WIDTH >= EXPANDED_BREAKPOINT;
        self.app_bar(ctx, &snapshot, expanded);
        self.rail(ctx);
        let colors = theme::colors(ctx);
        let margin = Margin { left: 0, right: spacing::XL as i8, top: 0, bottom: spacing::XL as i8 };
        CentralPanel::default().frame(egui::Frame::new().fill(colors.surface).inner_margin(margin)).show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            match self.screen {
                Screen::Live => self.live.ui(ui, &self.session, &snapshot, expanded),
                other => offline_screen(ui, rect, other),
            }
        });

        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        drop_overlay(ctx, ctx.content_rect(), "Drop to use this face", "The photo becomes the source face", hovering);
        if let Some(capture) = &mut self.capture {
            capture.update(ctx, &snapshot.phase);
        }
    }
}
