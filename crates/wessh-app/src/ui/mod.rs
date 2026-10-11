//! Desktop UI (egui), styled after Material Design 3. Talks to the live worker only through
//! [`LiveSession`] and to the offline (photo / video) worker only through [`JobRunner`].
//!
//! - [`theme`]: HCT colour scheme, type scale, tokens and the egui style mapping.
//! - [`components`]: reusable MD3 widgets.
//! - [`screens`]: the Live / Image / Video destinations, and the first-run onboarding shown
//!   instead of them until everything is installed.

mod capture;
mod components;
mod file_picker;
mod icons;
mod screens;
mod task;
mod theme;
mod tools;

use crate::cli::Options;
use crate::jobs::JobRunner;
use crate::live::{EngineOptions, LiveSession, Phase, Snapshot};
use components::navigation::{RAIL_WIDTH, TOP_APP_BAR_HEIGHT};
use components::{Chip, IconButton, drop_overlay, navigation_rail, top_app_bar};
use eframe::egui::{self, CentralPanel, Context, Key, Margin, SidePanel, TopBottomPanel};
use file_picker::{IMAGE_EXTENSIONS, has_extension};
use screens::onboarding::{Next, Onboarding};
use screens::{ImageScreen, LiveScreen, Screen, VideoScreen};
use std::path::PathBuf;
use theme::{Brightness, ThemeConfig, spacing};

const INITIAL_WINDOW: [f32; 2] = [1320.0, 840.0];
const MIN_WINDOW: [f32; 2] = [560.0, 480.0];
/// Content width from which the settings sheet docks beside the preview
/// (MD3 "expanded" window class starts at 840 dp).
const EXPANDED_BREAKPOINT: f32 = 840.0;
const APP_NAME: &str = "wessh";

/// Open the desktop app.
pub fn run(options: Options) -> anyhow::Result<()> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(INITIAL_WINDOW)
            .with_min_inner_size(MIN_WINDOW)
            .with_title(APP_NAME)
            .with_app_id("wessh")
            .with_icon(window_icon()),
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(APP_NAME, native, Box::new(move |cc| Ok(Box::new(Shell::new(&cc.egui_ctx, options)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// The window: onboarding until everything is installed, then the app. The engine (and so the
/// models) only loads once the app starts.
struct Shell {
    options: Options,
    onboarding: Option<Box<Onboarding>>,
    app: Option<Box<App>>,
    theme: ThemeConfig,
    /// Design review capture, while onboarding shows (the app takes it over afterwards).
    capture: Option<capture::Capture>,
}

impl Shell {
    fn new(ctx: &Context, options: Options) -> Self {
        let dev = capture::DevOptions::from_env();
        let brightness = if dev.light { Brightness::Light } else { Brightness::Dark };
        let theme = ThemeConfig { brightness, ..ThemeConfig::default() };
        theme::install_fonts(ctx);
        theme::apply(ctx, theme);
        let onboarding = Onboarding::check(ctx, &options, dev.onboarding).map(Box::new);
        let mut shell = Self { options, onboarding, app: None, theme, capture: None };
        if shell.onboarding.is_some() {
            shell.capture = dev.capture;
        } else {
            shell.open_app(ctx);
        }
        shell
    }

    /// Initialise the engine and show the app.
    fn open_app(&mut self, ctx: &Context) {
        self.onboarding = None;
        let engine = EngineOptions { runtime: self.options.ort.clone(), device: self.options.device };
        let repaint = ctx.clone();
        let session = LiveSession::spawn(engine.clone(), self.options.settings(), move || repaint.request_repaint());
        let repaint = ctx.clone();
        let jobs = JobRunner::spawn(engine, move || repaint.request_repaint());
        self.app = Some(Box::new(App::new(ctx, session, jobs, self.theme)));
    }
}

impl eframe::App for Shell {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        if let Some(onboarding) = &mut self.onboarding {
            if onboarding.ui(ctx) == Next::OpenApp {
                self.open_app(ctx);
            } else if let Some(capture) = &mut self.capture {
                capture.update(ctx, &Phase::Ready);
            }
            return;
        }
        if let Some(app) = &mut self.app {
            app.update(ctx, frame);
        }
    }
}

struct App {
    session: LiveSession,
    jobs: JobRunner,
    theme: ThemeConfig,
    screen: Screen,
    live: LiveScreen,
    image: ImageScreen,
    video: VideoScreen,
    capture: Option<capture::Capture>,
}

impl App {
    fn new(ctx: &Context, session: LiveSession, jobs: JobRunner, theme: ThemeConfig) -> Self {
        let dev = capture::DevOptions::from_env();
        let mut app = Self {
            session,
            jobs,
            theme,
            screen: dev.screen,
            live: LiveScreen::default(),
            image: ImageScreen::default(),
            video: VideoScreen::default(),
            capture: dev.capture,
        };
        if let Some(source) = dev.source {
            app.live.request_source(ctx, &app.session, &Snapshot::default(), source);
        }
        if let Some(target) = dev.target {
            app.open_target(ctx, target);
        }
        if let Some(output) = dev.output {
            app.video.set_output(&output);
        }
        if dev.loopback_missing {
            app.live.pretend_loopback_missing();
        }
        if dev.render {
            app.video.render_when_ready();
        }
        app
    }

    fn toggle_brightness(&mut self, ctx: &Context) {
        self.theme.brightness = self.theme.brightness.toggled();
        theme::apply(ctx, self.theme);
    }

    /// Give a photo or video to the screen that works on it.
    fn open_target(&mut self, ctx: &Context, path: PathBuf) {
        match self.screen {
            Screen::Video => self.video.set_target(ctx, path),
            Screen::Image | Screen::Live => self.image.set_target(ctx, path),
        }
    }

    /// Live: dropped photos become the source face. Image / Video: dropped files become the
    /// target, except photos dropped on the Video screen, which become the source face.
    fn handle_drops(&mut self, ctx: &Context, snapshot: &Snapshot) {
        let dropped = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        let Some(path) = dropped else { return };
        match self.screen {
            Screen::Image => self.image.set_target(ctx, path),
            Screen::Video if !has_extension(&path, &IMAGE_EXTENSIONS) => self.video.set_target(ctx, path),
            Screen::Live | Screen::Video => self.live.request_source(ctx, &self.session, snapshot, path),
        }
    }

    /// What the drop overlay promises on the current screen.
    fn drop_copy(&self) -> (&'static str, &'static str) {
        match self.screen {
            Screen::Live => ("Drop to use this face", "The photo becomes the source face"),
            Screen::Image => ("Drop to swap this photo", "The source face is put into it"),
            Screen::Video => ("Drop a video to render", "Or a photo to use as the source face"),
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
            top_app_bar(ui, APP_NAME, |ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                let (icon, tip) = if colors.is_dark() { (icons::LIGHT_MODE, "Light theme") } else { (icons::DARK_MODE, "Dark theme") };
                toggle_theme = ui.add(IconButton::new(icon, tip)).clicked();
                if !expanded {
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

impl App {
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
            let (session, jobs) = (&self.session, &self.jobs);
            match self.screen {
                Screen::Live => self.live.ui(ui, session, &snapshot, expanded),
                Screen::Image => {
                    self.live.show(ui, session, &snapshot, expanded, true, |ui, rect, frame| self.image.ui(ui, rect, frame, jobs));
                }
                Screen::Video => {
                    self.live.show(ui, session, &snapshot, expanded, true, |ui, rect, frame| self.video.ui(ui, rect, frame, jobs));
                }
            }
        });

        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let (title, body) = self.drop_copy();
        drop_overlay(ctx, ctx.content_rect(), title, body, hovering);
        if let Some(capture) = &mut self.capture {
            capture.update(ctx, &snapshot.phase);
        }
    }
}

/// The app icon for the title bar, taskbar and window switcher.
fn window_icon() -> egui::IconData {
    let image = image::load_from_memory_with_format(include_bytes!("../../assets/icon/wessh-256.png"), image::ImageFormat::Png)
        .expect("the embedded icon is a valid PNG")
        .into_rgba8();
    egui::IconData { width: image.width(), height: image.height(), rgba: image.into_raw() }
}
