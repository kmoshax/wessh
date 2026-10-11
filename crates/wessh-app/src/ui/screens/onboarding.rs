//! First-run onboarding: shown instead of the app while the runtime or a curated model is
//! missing. Welcome → Getting ready (download everything, with pause / resume / retry) → Done.
//!
//! Nothing is chosen here: the hardware is detected, TensorRT is included on NVIDIA GPUs and the
//! models are the curated set ([`wessh_core::assets::required`]). The engine is only initialised
//! once this flow has finished.

use super::layout::{centered, column};
use crate::cli::Options;
use crate::install::progress::{human_duration, human_rate, human_size, time_left};
use crate::install::{self, Host, Installer, Job, Kind, Message, Outcome, Plan, Request, Stage, platform, sources};
use crate::ui::components::{Button, CircularProgress, LinearProgress, paint};
use crate::ui::icons;
use crate::ui::theme::{self, ColorScheme, TypeRole, shape, size, spacing};
use eframe::egui::{
    Align, Align2, Color32, Context, Frame, Key, Label, Layout, Margin, Rect, ScrollArea, Sense, Stroke, Ui, UiBuilder, pos2, vec2,
};
use wessh_core::Device;
use wessh_core::assets;
use wessh_core::runtime::locate;

/// Width of the welcome and done pages' content (MD3 dialog-like reading width).
const NARROW_WIDTH: f32 = 560.0;
/// Width of the download page, which lists every component.
const WIDE_WIDTH: f32 = 640.0;
const HERO_CIRCLE: f32 = 72.0;
const HERO_ICON: f32 = 36.0;
/// Summary rows on the welcome page and component rows on the download page.
const ROW_HEIGHT: f32 = 56.0;
const ROW_ICON_CIRCLE: f32 = 40.0;
const STATUS_SPINNER: f32 = 20.0;
/// Step indicator: circle diameter and the connector drawn between steps.
const STEP_DOT: f32 = 24.0;
const STEP_CONNECTOR: f32 = 40.0;
const STEP_LABELS: [&str; 3] = ["Welcome", "Download", "Ready"];
const PERCENT: f32 = 100.0;

/// A page of the flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Welcome,
    Download,
    Done,
}

impl Step {
    /// `WESSH_UI_ONBOARDING` values.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "welcome" => Some(Self::Welcome),
            "download" => Some(Self::Download),
            "done" => Some(Self::Done),
            _ => None,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Welcome => 0,
            Self::Download => 1,
            Self::Done => 2,
        }
    }
}

/// One component as the download page shows it.
struct Row {
    label: String,
    version: String,
    kind: Kind,
    size: Option<u64>,
    /// Bytes on disk.
    done: u64,
    stage: Stage,
}

impl Row {
    /// Bytes counted as done towards the overall bar (everything once installed).
    fn progress(&self) -> u64 {
        match self.size {
            Some(size) if self.stage == Stage::Done => size,
            Some(size) => self.done.min(size),
            None => self.done,
        }
    }
}

/// What the app should do after drawing a frame of the flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    Stay,
    /// Everything is installed and the user moved on: start the app.
    OpenApp,
}

/// State of the onboarding flow.
pub struct Onboarding {
    step: Step,
    /// The plan while no run owns it.
    plan: Option<Plan>,
    rows: Vec<Row>,
    /// Plan indices of the components this flow installs (those missing when it started).
    shown: Vec<usize>,
    run: Option<(Installer, Job)>,
    /// "Get started" was pressed while the sizes were still being looked up.
    start_after_resolve: bool,
    pausing: bool,
    paused: bool,
    failure: Option<String>,
    rate: Option<f64>,
    /// Nothing was installed before: a first run rather than an update.
    first_run: bool,
    gpu: Option<String>,
    flavour: platform::Flavour,
    /// Design review: pause the download at this time (`WESSH_UI_PAUSE_AT`).
    pause_at: Option<f64>,
}

/// What onboarding installs for these options: the detected runtime (CPU only with
/// `--device cpu`), TensorRT on NVIDIA unless the device rules it out, and the curated models.
fn request(options: &Options) -> Request {
    let runtime = locate::runtime_library(options.ort.as_deref()).ok();
    let external_runtime = runtime.is_some_and(|path| path.parent() != Some(locate::default_runtime_dir().as_path()));
    Request {
        tensorrt: matches!(options.device, Device::Auto | Device::Tensorrt),
        ort_version: sources::ORT_VERSION.to_owned(),
        force: false,
        models: assets::required(&options.settings()).to_vec(),
        external_runtime,
        ffmpeg: false,
    }
}

impl Onboarding {
    /// The flow, if something is missing (or `forced` asks for a page anyway, for design review).
    pub fn check(ctx: &Context, options: &Options, forced: Option<Step>) -> Option<Self> {
        let flavour = (options.device == Device::Cpu).then_some(platform::Flavour::Cpu);
        let host = match Host::detect(flavour) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("skipping setup: {e:#}");
                return None;
            }
        };
        let plan = Plan::check(host, &request(options));
        if plan.is_complete() && forced.is_none() {
            return None;
        }
        Some(Self::new(ctx, plan, forced))
    }

    fn new(ctx: &Context, plan: Plan, forced: Option<Step>) -> Self {
        let rows = plan
            .items
            .iter()
            .map(|item| Row {
                label: item.label.clone(),
                version: item.version.clone(),
                kind: item.kind,
                size: item.size(),
                done: item.partial,
                stage: if item.installed { Stage::Done } else { Stage::Queued },
            })
            .collect();
        let resuming = plan.has_partial_downloads();
        let mut onboarding = Self {
            step: forced.unwrap_or(if resuming { Step::Download } else { Step::Welcome }),
            shown: plan.pending().collect(),
            rows,
            first_run: plan.items.iter().all(|i| !i.installed),
            gpu: plan.host.gpu_name().map(str::to_owned),
            flavour: plan.host.choice.flavour,
            plan: Some(plan),
            run: None,
            start_after_resolve: false,
            pausing: false,
            paused: false,
            failure: None,
            rate: None,
            pause_at: std::env::var("WESSH_UI_PAUSE_AT").ok().and_then(|s| s.parse().ok()),
        };
        match onboarding.step {
            Step::Welcome => onboarding.start(ctx, Job::Resolve),
            Step::Download => onboarding.start(ctx, Job::Install),
            Step::Done => {}
        }
        onboarding
    }

    fn start(&mut self, ctx: &Context, job: Job) {
        let Some(plan) = self.plan.take() else { return };
        let ctx = ctx.clone();
        self.run = Some((Installer::spawn(plan, job, move || ctx.request_repaint()), job));
        self.failure = None;
        self.paused = false;
        self.pausing = false;
    }

    fn pause(&mut self) {
        if let Some((installer, _)) = &self.run {
            installer.cancel();
            self.pausing = true;
        }
    }

    /// Download (or resume) now, or as soon as the size lookup has finished.
    fn download(&mut self, ctx: &Context) {
        self.step = Step::Download;
        if self.run.is_some() {
            self.start_after_resolve = true;
        } else {
            self.start(ctx, Job::Install);
        }
    }

    /// Apply what the installer reported.
    fn poll(&mut self, ctx: &Context) {
        let Some((installer, job)) = &self.run else { return };
        let job = *job;
        let messages: Vec<Message> = installer.poll().collect();
        for message in messages {
            match message {
                Message::Event(event) => self.apply(event),
                Message::Finished { plan, outcome } => {
                    self.run = None;
                    self.rate = None;
                    self.pausing = false;
                    let complete = plan.is_complete();
                    self.plan = Some(*plan);
                    self.finished(ctx, job, outcome, complete);
                }
            }
        }
    }

    fn apply(&mut self, event: install::Event) {
        match event {
            install::Event::Stage { item, stage } => self.rows[item].stage = stage,
            install::Event::Resolved { item, version, size, partial } => {
                let row = &mut self.rows[item];
                (row.version, row.size, row.done) = (version, size, partial);
            }
            install::Event::Bytes { item, done, bytes_per_sec } => {
                self.rows[item].done = done;
                // A new file has no rate of its own yet; the last one is the best estimate.
                self.rate = bytes_per_sec.or(self.rate);
            }
        }
    }

    fn finished(&mut self, ctx: &Context, job: Job, outcome: Outcome, complete: bool) {
        match (job, outcome) {
            (Job::Resolve, Outcome::Done | Outcome::Failed(_)) if self.start_after_resolve => {
                // A failed lookup is retried by the install run, which reports it properly.
                self.start_after_resolve = false;
                self.start(ctx, Job::Install);
            }
            (Job::Install, Outcome::Done) if complete => {
                self.step = Step::Done;
            }
            (_, Outcome::Cancelled) => self.paused = true,
            (Job::Install, Outcome::Failed(error)) => self.failure = Some(error),
            _ => {}
        }
        // Queued rows of a stopped run go back to waiting.
        for row in &mut self.rows {
            if matches!(row.stage, Stage::Resolving | Stage::Downloading | Stage::Verifying | Stage::Extracting) {
                row.stage = Stage::Queued;
            }
        }
    }

    /// Bytes done and total over the components this flow installs (known sizes only).
    fn totals(&self) -> (u64, u64) {
        self.shown
            .iter()
            .map(|&i| &self.rows[i])
            .fold((0, 0), |(done, total), row| (done + row.progress(), total + row.size.unwrap_or_default()))
    }

    /// Download size of the shown components of `kinds`, if every one is known.
    fn size_of(&self, kinds: &[Kind]) -> Option<u64> {
        self.shown.iter().map(|&i| &self.rows[i]).filter(|r| kinds.contains(&r.kind)).map(|r| r.size).sum()
    }

    fn has(&self, kind: Kind) -> bool {
        self.shown.iter().any(|&i| self.rows[i].kind == kind)
    }

    fn resolving(&self) -> bool {
        self.run.as_ref().is_some_and(|(_, job)| *job == Job::Resolve)
    }

    /// Draw the flow over the whole window.
    pub fn ui(&mut self, ctx: &Context) -> Next {
        self.poll(ctx);
        let now = ctx.input(|i| i.time);
        if self.pause_at.is_some_and(|at| now >= at) && self.step == Step::Download && self.run.is_some() {
            self.pause_at = None;
            self.pause();
        }
        let colors = theme::colors(ctx);
        let mut next = Next::Stay;
        let frame = Frame::new().fill(colors.surface).inner_margin(Margin::same(spacing::XL as i8));
        eframe::egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            let area = ui.max_rect();
            let header = Rect::from_min_size(area.min, vec2(area.width(), STEP_DOT + spacing::XL));
            step_indicator(ui, header, self.step, &colors);
            let body = Rect::from_min_max(pos2(area.left(), header.bottom() + spacing::LG), area.max);
            match self.step {
                Step::Welcome => centered(ui, column(body, NARROW_WIDTH), "onboarding-welcome", |ui| self.welcome(ui, &colors)),
                Step::Download => {
                    let rect = column(body, WIDE_WIDTH);
                    ui.scope_builder(UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), |ui| {
                        self.downloading(ui, &colors);
                    });
                }
                Step::Done => centered(ui, column(body, NARROW_WIDTH), "onboarding-done", |ui| {
                    if Self::done(ui, &colors) {
                        next = Next::OpenApp;
                    }
                }),
            }
        });
        next
    }

    // ── Welcome ──────────────────────────────────────────────────────────────────

    fn welcome(&mut self, ui: &mut Ui, colors: &ColorScheme) {
        hero(ui, icons::FACE, colors.primary_container, colors.on_primary_container);
        ui.add_space(spacing::LG);
        let title = if self.first_run { "Welcome to wessh" } else { "Finish setting up wessh" };
        ui.label(TypeRole::HeadlineMedium.rich(title, colors.on_surface));
        ui.add_space(spacing::SM);
        let body = if self.first_run {
            "Swap your face for anyone's, live on your camera, in photos and in videos."
        } else {
            "A few components are still missing. wessh downloads them once, then works offline."
        };
        wrapped(ui, TypeRole::BodyLarge, body, colors.on_surface_variant, Align::Center);
        ui.add_space(spacing::XL);

        let checking = self.resolving();
        let card =
            Frame::new().fill(colors.surface_container_low).corner_radius(shape::LARGE).inner_margin(Margin::same(spacing::LG as i8));
        card.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.with_layout(Layout::top_down(Align::Min), |ui| self.summary(ui, colors, checking));
        });

        ui.add_space(spacing::LG);
        let consent = "Only swap faces you have permission to use, and never to deceive or impersonate anyone.";
        note(ui, icons::VERIFIED_USER, consent, colors.on_surface_variant, colors);
        ui.add_space(spacing::XL);
        let start = ui.add(Button::filled("Get started").icon(icons::DOWNLOAD)).clicked();
        if start || ui.input(|i| i.key_pressed(Key::Enter)) {
            self.download(ui.ctx());
        }
    }

    /// What will be downloaded, grouped, with sizes once known.
    fn summary(&self, ui: &mut Ui, colors: &ColorScheme, checking: bool) {
        let size = |sizes: Option<u64>| match sizes {
            Some(bytes) => human_size(bytes),
            None if checking => "Checking…".to_owned(),
            None => "—".to_owned(),
        };
        {
            ui.label(TypeRole::TitleSmall.rich("wessh will download and set up", colors.on_surface_variant));
            ui.add_space(spacing::XS);
            if self.has(Kind::Runtime) || self.has(Kind::CudaLibrary) {
                let title = match (&self.gpu, self.flavour.cuda_major()) {
                    (Some(gpu), Some(_)) => format!("Engine for your {gpu}"),
                    _ => "Engine for your processor".to_owned(),
                };
                let detail = match self.flavour.cuda_major() {
                    Some(major) => format!("ONNX Runtime with NVIDIA CUDA {major} and cuDNN"),
                    None => "ONNX Runtime".to_owned(),
                };
                summary_row(ui, icons::MEMORY, &title, &detail, &size(self.size_of(&[Kind::Runtime, Kind::CudaLibrary])));
            }
            if self.has(Kind::TensorRt) {
                let detail = "NVIDIA's fastest inference engine for your GPU";
                summary_row(ui, icons::BOLT, "TensorRT acceleration", detail, &size(self.size_of(&[Kind::TensorRt])));
            }
            if self.has(Kind::Model) {
                let count = self.shown.iter().filter(|&&i| self.rows[i].kind == Kind::Model).count();
                let detail = format!("{count} models for detection, swapping, enhancement and masks");
                summary_row(ui, icons::FACE, "Face models", &detail, &size(self.size_of(&[Kind::Model])));
            }
            ui.add_space(spacing::SM);
            divider(ui, colors.outline_variant);
            ui.add_space(spacing::MD);
            ui.horizontal(|ui| {
                ui.label(TypeRole::TitleMedium.rich("Total", colors.on_surface));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let total = self.size_of(&[Kind::Runtime, Kind::CudaLibrary, Kind::TensorRt, Kind::Model]);
                    ui.label(TypeRole::TitleMedium.rich(size(total), colors.on_surface));
                });
            });
            ui.add_space(spacing::XXS);
            let note = if !checking && self.size_of(&[Kind::Model]).is_none() {
                "Couldn't check the sizes right now. You can still start; wessh will keep trying."
            } else {
                "If the download is interrupted, it continues where it stopped."
            };
            wrapped(ui, TypeRole::BodyMedium, note, colors.on_surface_variant, Align::Min);
        }
    }

    // ── Getting ready ────────────────────────────────────────────────────────────

    fn downloading(&mut self, ui: &mut Ui, colors: &ColorScheme) {
        ui.label(TypeRole::HeadlineSmall.rich("Getting ready", colors.on_surface));
        ui.add_space(spacing::XS);
        let body = "wessh is downloading what it needs. You can pause at any time, even quit, and continue later.";
        wrapped(ui, TypeRole::BodyLarge, body, colors.on_surface_variant, Align::Min);
        ui.add_space(spacing::XL);

        let mut action = None;
        let card =
            Frame::new().fill(colors.surface_container_low).corner_radius(shape::LARGE).inner_margin(Margin::same(spacing::XL as i8));
        card.show(ui, |ui| {
            ui.set_width(ui.available_width());
            action = self.overall(ui, colors);
        });
        if let Some(error) = &self.failure {
            ui.add_space(spacing::LG);
            error_card(ui, error, colors);
        }
        match action {
            Some(Action::Pause) => self.pause(),
            Some(Action::Resume | Action::Retry) => self.download(ui.ctx()),
            None => {}
        }

        ui.add_space(spacing::LG);
        let list = Frame::new()
            .stroke(Stroke::new(size::STROKE, colors.outline_variant))
            .corner_radius(shape::LARGE)
            .inner_margin(Margin::symmetric(spacing::LG as i8, spacing::XS as i8));
        list.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ScrollArea::vertical().auto_shrink([false, true]).show(ui, |ui| {
                for &index in &self.shown {
                    component_row(ui, &self.rows[index], colors);
                }
            });
        });
    }

    /// Overall bar, amounts, speed and the pause / resume button.
    fn overall(&self, ui: &mut Ui, colors: &ColorScheme) -> Option<Action> {
        let (done, total) = self.totals();
        let fraction = if total == 0 { 0.0 } else { done as f32 / total as f32 };
        let sized = total > 0 && self.shown.iter().all(|&i| self.rows[i].size.is_some());
        ui.horizontal(|ui| {
            let amount = if sized {
                format!("{} of {}", human_size(done), human_size(total))
            } else if self.run.is_some() {
                "Checking sizes…".to_owned()
            } else {
                format!("{} downloaded", human_size(done))
            };
            ui.label(TypeRole::TitleLarge.rich(amount, colors.on_surface));
            if sized {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(TypeRole::TitleLarge.rich(format!("{:.0}%", (fraction * PERCENT).floor()), colors.primary));
                });
            }
        });
        ui.add_space(spacing::MD);
        ui.add(if sized || self.run.is_none() { LinearProgress::determinate(fraction) } else { LinearProgress::indeterminate() });
        ui.add_space(spacing::MD);
        let running = self.run.is_some();
        let status = if self.pausing {
            "Pausing…".to_owned()
        } else if self.failure.is_some() {
            "Stopped. Everything downloaded so far is kept.".to_owned()
        } else if self.paused || !running {
            "Paused. Everything downloaded so far is kept.".to_owned()
        } else if let Some(rate) = self.rate {
            match time_left(total.saturating_sub(done), Some(rate)) {
                Some(left) => format!("{}  ·  About {} left", human_rate(rate), human_duration(left)),
                None => human_rate(rate),
            }
        } else if self.resolving() || self.shown.iter().any(|&i| self.rows[i].stage == Stage::Resolving) {
            "Checking what to download…".to_owned()
        } else {
            "Connecting…".to_owned()
        };
        let mut action = None;
        ui.horizontal(|ui| {
            ui.set_min_height(size::TARGET);
            ui.label(TypeRole::BodyMedium.rich(status, colors.on_surface_variant));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if running {
                    let enabled = !self.pausing;
                    if ui.add_enabled(enabled, Button::outlined("Pause").icon(icons::PAUSE)).clicked() {
                        action = Some(Action::Pause);
                    }
                } else if self.failure.is_some() {
                    if ui.add(Button::filled("Try again").icon(icons::REFRESH)).clicked() {
                        action = Some(Action::Retry);
                    }
                } else if ui.add(Button::filled("Resume").icon(icons::PLAY_ARROW)).clicked() {
                    action = Some(Action::Resume);
                }
            });
        });
        action
    }

    // ── Done ─────────────────────────────────────────────────────────────────────

    /// Returns `true` when the user moves on to the app.
    fn done(ui: &mut Ui, colors: &ColorScheme) -> bool {
        hero(ui, icons::TASK_ALT, colors.primary_container, colors.on_primary_container);
        ui.add_space(spacing::LG);
        ui.label(TypeRole::HeadlineMedium.rich("You're all set", colors.on_surface));
        ui.add_space(spacing::SM);
        let body = "wessh is ready. Pick a photo of a face, then start your camera.";
        wrapped(ui, TypeRole::BodyLarge, body, colors.on_surface_variant, Align::Center);
        ui.add_space(spacing::XL);
        let open = ui.add(Button::filled("Start using wessh").icon(icons::ARROW_FORWARD)).clicked();
        open || ui.input(|i| i.key_pressed(Key::Enter))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Pause,
    Resume,
    Retry,
}

/// Large icon in a tonal circle at the top of a page.
fn hero(ui: &mut Ui, icon: char, container: Color32, content: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(HERO_CIRCLE, HERO_CIRCLE), Sense::hover());
    ui.painter().circle_filled(rect.center(), HERO_CIRCLE / 2.0, container);
    paint::icon(ui.painter(), rect.center(), icon, HERO_ICON, content, false);
}

/// Text wrapped to the available width, aligned as `align`.
fn wrapped(ui: &mut Ui, role: TypeRole, text: &str, color: Color32, align: Align) {
    let mut job = role.job(text, color, Some(ui.available_width()));
    job.halign = align;
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    ui.add(Label::new(galley).selectable(false));
}

/// A leading icon and wrapped text.
fn note(ui: &mut Ui, icon: char, text: &str, color: Color32, colors: &ColorScheme) {
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = spacing::MD;
        let (rect, _) = ui.allocate_exact_size(vec2(size::ICON, size::ICON), Sense::hover());
        paint::icon(ui.painter(), rect.center(), icon, size::ICON, colors.primary, false);
        wrapped(ui, TypeRole::BodyMedium, text, color, Align::Min);
    });
}

fn divider(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), size::STROKE), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, color);
}

/// Icon in a tonal circle, two lines of text and a trailing value.
fn summary_row(ui: &mut Ui, icon: char, title: &str, detail: &str, trailing: &str) {
    let colors = theme::colors(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::hover());
    let circle = pos2(rect.left() + ROW_ICON_CIRCLE / 2.0, rect.center().y);
    ui.painter().circle_filled(circle, ROW_ICON_CIRCLE / 2.0, colors.secondary_container);
    paint::icon(ui.painter(), circle, icon, size::ICON, colors.on_secondary_container, false);
    let trailing_rect = paint::text(
        ui,
        pos2(rect.right(), rect.center().y),
        Align2::RIGHT_CENTER,
        TypeRole::LabelLarge,
        trailing,
        colors.on_surface_variant,
    );
    let x = rect.left() + ROW_ICON_CIRCLE + spacing::LG;
    let width = trailing_rect.left() - spacing::LG - x;
    two_lines(ui, pos2(x, rect.center().y), width, title, detail, &colors);
}

/// Headline and supporting text centred vertically on `left_center`, each cut to one line.
fn two_lines(ui: &Ui, left_center: eframe::egui::Pos2, width: f32, title: &str, detail: &str, colors: &ColorScheme) {
    let line = |role: TypeRole| role.style().line_height;
    let top = left_center.y - f32::midpoint(line(TypeRole::BodyLarge), line(TypeRole::BodyMedium));
    paint::text_truncated(ui, pos2(left_center.x, top), Align2::LEFT_TOP, TypeRole::BodyLarge, title, colors.on_surface, width);
    let below = top + line(TypeRole::BodyLarge);
    paint::text_truncated(ui, pos2(left_center.x, below), Align2::LEFT_TOP, TypeRole::BodyMedium, detail, colors.on_surface_variant, width);
}

/// One component on the download page: state icon, name, amounts and percentage.
fn component_row(ui: &mut Ui, row: &Row, colors: &ColorScheme) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::hover());
    let icon_center = pos2(rect.left() + size::ICON / 2.0, rect.center().y);
    match &row.stage {
        Stage::Done => paint::icon(ui.painter(), icon_center, icons::CHECK_CIRCLE, size::ICON, colors.primary, true),
        Stage::Downloading => paint::icon(ui.painter(), icon_center, icons::DOWNLOAD, size::ICON, colors.primary, false),
        Stage::Failed(_) => paint::icon(ui.painter(), icon_center, icons::ERROR, size::ICON, colors.error, true),
        Stage::Queued if row.done > 0 => paint::icon(ui.painter(), icon_center, icons::PAUSE, size::ICON, colors.on_surface_variant, false),
        Stage::Queued => paint::icon(ui.painter(), icon_center, icons::SCHEDULE, size::ICON, colors.outline, false),
        Stage::Resolving | Stage::Verifying | Stage::Extracting => {
            let spinner = Rect::from_center_size(icon_center, vec2(STATUS_SPINNER, STATUS_SPINNER));
            ui.put(spinner, CircularProgress::new(STATUS_SPINNER));
        }
    }
    let size_text = row.size.map(human_size);
    let detail = match (&row.stage, &size_text) {
        (Stage::Downloading, Some(total)) => format!("{} of {total}", human_size(row.done)),
        (Stage::Downloading, None) => human_size(row.done),
        (Stage::Resolving, _) => "Looking up the latest version…".to_owned(),
        (Stage::Verifying, _) => "Checking the download…".to_owned(),
        (Stage::Extracting, _) => "Installing…".to_owned(),
        (Stage::Done, _) => "Installed".to_owned(),
        (Stage::Failed(_), _) => "Couldn't download".to_owned(),
        (Stage::Queued, Some(total)) if row.done > 0 => format!("{} of {total} · paused", human_size(row.done)),
        (Stage::Queued, Some(total)) if row.version.is_empty() => total.clone(),
        (Stage::Queued, Some(total)) => format!("Version {} · {total}", row.version),
        (Stage::Queued, None) => "Waiting".to_owned(),
    };
    let percent = match (&row.stage, row.size) {
        (Stage::Downloading, Some(total)) if total > 0 => Some(format!("{:.0}%", (row.done as f32 / total as f32 * PERCENT).floor())),
        (Stage::Done, _) => size_text,
        _ => None,
    };
    let trailing = percent.map_or(rect.right(), |text| {
        let color = if row.stage == Stage::Done { colors.on_surface_variant } else { colors.primary };
        paint::text(ui, pos2(rect.right(), rect.center().y), Align2::RIGHT_CENTER, TypeRole::LabelLarge, &text, color).left()
    });
    let title = &row.label;
    let x = rect.left() + size::ICON + spacing::LG;
    two_lines(ui, pos2(x, rect.center().y), trailing - spacing::LG - x, title, &detail, colors);
}

/// What went wrong, calmly: interrupted downloads are common and nothing is lost. "Try again"
/// sits with the progress, where "Resume" otherwise is.
fn error_card(ui: &mut Ui, error: &str, colors: &ColorScheme) {
    let frame = Frame::new()
        .fill(colors.surface_container_low)
        .stroke(Stroke::new(size::STROKE, colors.error))
        .corner_radius(shape::LARGE)
        .inner_margin(Margin::same(spacing::LG as i8));
    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = spacing::LG;
            let (rect, _) = ui.allocate_exact_size(vec2(size::ICON, size::ICON), Sense::hover());
            paint::icon(ui.painter(), rect.center(), icons::CLOUD_OFF, size::ICON, colors.error, false);
            ui.vertical(|ui| {
                ui.label(TypeRole::TitleMedium.rich("The download stopped", colors.on_surface));
                ui.add_space(spacing::XS);
                let advice = "Check your internet connection, then try again. Nothing downloaded so far is lost.";
                wrapped(ui, TypeRole::BodyMedium, advice, colors.on_surface_variant, Align::Min);
                ui.add_space(spacing::SM);
                wrapped(ui, TypeRole::BodySmall, &short_error(error), colors.outline, Align::Min);
            });
        });
    });
}

/// Longest technical detail shown; the full error goes to stderr.
const ERROR_DETAIL_CHARS: usize = 240;

/// The technical detail of an error, shortened for the banner.
fn short_error(error: &str) -> String {
    let text: String = error.chars().take(ERROR_DETAIL_CHARS).collect();
    if text.len() < error.len() { format!("{text}…") } else { text }
}

/// The three steps across the top, the current one highlighted.
fn step_indicator(ui: &Ui, rect: Rect, step: Step, colors: &ColorScheme) {
    let label_width = |label: &str| paint::text_width(ui, TypeRole::LabelLarge, label);
    let widths: Vec<f32> = STEP_LABELS.iter().map(|label| STEP_DOT + spacing::SM + label_width(label)).collect();
    let total =
        widths.iter().sum::<f32>() + STEP_CONNECTOR * (STEP_LABELS.len() - 1) as f32 + 2.0 * spacing::SM * (STEP_LABELS.len() - 1) as f32;
    let y = rect.top() + STEP_DOT / 2.0;
    let mut x = rect.center().x - total / 2.0;
    for (index, (label, width)) in STEP_LABELS.iter().zip(&widths).enumerate() {
        let center = pos2(x + STEP_DOT / 2.0, y);
        let (reached, current) = (index <= step.index(), index == step.index());
        if reached {
            ui.painter().circle_filled(center, STEP_DOT / 2.0, colors.primary);
            if index < step.index() || step == Step::Done {
                paint::icon(ui.painter(), center, icons::CHECK, size::ICON_SMALL, colors.on_primary, false);
            } else {
                paint::text(ui, center, Align2::CENTER_CENTER, TypeRole::LabelMedium, &(index + 1).to_string(), colors.on_primary);
            }
        } else {
            ui.painter().circle_stroke(center, STEP_DOT / 2.0 - size::STROKE / 2.0, Stroke::new(size::STROKE, colors.outline));
            paint::text(ui, center, Align2::CENTER_CENTER, TypeRole::LabelMedium, &(index + 1).to_string(), colors.on_surface_variant);
        }
        let color = if current { colors.on_surface } else { colors.on_surface_variant };
        paint::text(ui, pos2(x + STEP_DOT + spacing::SM, y), Align2::LEFT_CENTER, TypeRole::LabelLarge, label, color);
        x += width + spacing::SM;
        if index + 1 < STEP_LABELS.len() {
            let line = if index < step.index() { colors.primary } else { colors.outline_variant };
            ui.painter().line_segment([pos2(x, y), pos2(x + STEP_CONNECTOR, y)], Stroke::new(size::STROKE, line));
            x += STEP_CONNECTOR + spacing::SM;
        }
    }
}
