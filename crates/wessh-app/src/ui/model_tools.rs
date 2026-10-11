//! Optional models are downloaded when their feature is enabled, with progress and retry.

use super::components::{Button, LinearProgress};
use super::theme::{self, TypeRole, shape, spacing};
use crate::install::platform::{Choice, Flavour, Target};
use crate::install::progress::human_size;
use crate::install::{Event, Host, Installer, Job, Message, Outcome, Plan, Request};
use eframe::egui::{Context, Frame, Id, Margin, Modal};
use wessh_core::{Settings, assets};

#[derive(Default)]
pub struct ModelTools {
    pending: Option<Settings>,
    plan: Option<Plan>,
    run: Option<(Installer, Job)>,
    amounts: Vec<(u64, Option<u64>)>,
    error: Option<String>,
    cancelling: bool,
}

impl ModelTools {
    /// Leave the current settings running until the requested models are ready.
    pub fn request(&mut self, ctx: &Context, settings: Settings) {
        if self.pending.is_some() {
            return;
        }
        let target = match Target::host() {
            Ok(target) => target,
            Err(error) => {
                self.error = Some(error.to_string());
                self.pending = Some(settings);
                return;
            }
        };
        let host = Host { target, gpus: Vec::new(), choice: Choice { flavour: Flavour::Cpu, reason: String::new() } };
        let request = Request { models: assets::required(&settings), ffmpeg: false, ..Request::ffmpeg_only() };
        let plan = Plan::check(host, &request);
        self.amounts = plan.items.iter().map(|item| if item.installed { (0, Some(0)) } else { (item.partial, item.size()) }).collect();
        self.plan = Some(plan);
        self.pending = Some(settings);
        self.error = None;
        self.start(ctx, Job::Resolve);
    }

    fn start(&mut self, ctx: &Context, job: Job) {
        let Some(plan) = self.plan.take() else { return };
        let ctx = ctx.clone();
        self.run = Some((Installer::spawn(plan, job, move || ctx.request_repaint()), job));
        self.error = None;
    }

    fn poll(&mut self) -> Option<Settings> {
        let (installer, job) = self.run.as_ref()?;
        let job = *job;
        let messages: Vec<_> = installer.poll().collect();
        for message in messages {
            match message {
                Message::Event(Event::Resolved { item, size, partial, .. }) => self.amounts[item] = (partial, size),
                Message::Event(Event::Bytes { item, done, .. }) => self.amounts[item].0 = done,
                Message::Event(Event::Stage { .. }) => {}
                Message::Finished { plan, outcome } => {
                    self.run = None;
                    self.plan = Some(*plan);
                    if self.cancelling {
                        self.pending = None;
                        self.cancelling = false;
                    } else if job == Job::Install && outcome == Outcome::Done && self.plan.as_ref().is_some_and(Plan::is_complete) {
                        return self.pending.take();
                    } else if let Outcome::Failed(error) = outcome {
                        self.error = Some(error);
                    }
                }
            }
        }
        None
    }

    /// Return settings only after a successful installation; dismissing keeps the old settings.
    pub fn ui(&mut self, ctx: &Context) -> Option<Settings> {
        if let Some(settings) = self.poll() {
            return Some(settings);
        }
        let pending = self.pending.as_ref()?;
        let names = assets::required(pending)
            .into_iter()
            .filter(|asset| !asset.is_downloaded())
            .map(|asset| crate::install::model_label(&asset))
            .collect::<Vec<_>>();
        let colors = theme::colors(ctx);
        let frame =
            Frame::new().fill(colors.surface_container_high).corner_radius(shape::LARGE).inner_margin(Margin::same(spacing::XL as i8));
        let mut download = false;
        let mut cancel = false;
        let modal = Modal::new(Id::new("optional-models")).frame(frame).show(ctx, |ui| {
            ui.set_width(400.0_f32.min(ctx.content_rect().width() - 2.0 * spacing::XL));
            ui.label(TypeRole::TitleLarge.rich("Download for this feature", colors.on_surface));
            ui.add_space(spacing::MD);
            ui.label("This feature needs an additional model. It is downloaded once and kept for future use.");
            ui.add_space(spacing::SM);
            ui.label(names.join(", "));
            let total: Option<u64> = self.amounts.iter().map(|(_, size)| *size).sum();
            let done: u64 = self.amounts.iter().map(|(done, _)| done).sum();
            if let Some((_, job)) = &self.run {
                ui.add_space(spacing::MD);
                if self.cancelling {
                    ui.label("Stopping… Your download will be kept.");
                } else if *job == Job::Resolve {
                    ui.label("Checking download size…");
                } else {
                    ui.add(
                        total
                            .filter(|&total| total > 0)
                            .map_or_else(LinearProgress::indeterminate, |total| LinearProgress::determinate(done as f32 / total as f32)),
                    );
                    ui.label(format!("{} downloaded", human_size(done)));
                }
            } else if self.plan.is_some() {
                let size = total.map_or_else(|| "Not available yet".to_owned(), human_size);
                ui.label(format!("Download size: {size}"));
            }
            if let Some(error) = &self.error {
                ui.colored_label(colors.error, error);
            }
            ui.add_space(spacing::LG);
            ui.horizontal(|ui| {
                cancel = ui.add_enabled(!self.cancelling, Button::text("Cancel")).clicked();
                if self.run.is_none() && self.plan.is_some() {
                    let label = if self.error.is_some() { "Try again" } else { "Download" };
                    download = ui.add(Button::filled(label)).clicked();
                }
            });
        });
        if cancel || modal.should_close() {
            if let Some((installer, _)) = &self.run {
                installer.cancel();
                self.cancelling = true;
            } else {
                self.pending = None;
            }
        } else if download {
            self.start(ctx, Job::Install);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::platform::{Arch, Os};
    use std::time::{Duration, Instant};

    fn ready_download() -> ModelTools {
        let host = Host {
            target: Target { os: Os::Windows, arch: Arch::X86_64 },
            gpus: Vec::new(),
            choice: Choice { flavour: Flavour::Cpu, reason: String::new() },
        };
        let request = Request { ffmpeg: false, ..Request::ffmpeg_only() };
        ModelTools {
            pending: Some(Settings { occlusion: true, ..Settings::default() }),
            plan: Some(Plan::check(host, &request)),
            ..ModelTools::default()
        }
    }

    fn finish(tools: &mut ModelTools) -> Option<Settings> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while tools.run.is_some() {
            assert!(Instant::now() < deadline, "installer did not finish");
            if let Some(settings) = tools.poll() {
                return Some(settings);
            }
            std::thread::yield_now();
        }
        None
    }

    #[test]
    fn settings_are_applied_only_after_installation() {
        let mut tools = ready_download();
        tools.start(&Context::default(), Job::Resolve);
        assert!(finish(&mut tools).is_none());
        assert!(tools.pending.is_some());
        tools.start(&Context::default(), Job::Install);
        assert!(finish(&mut tools).unwrap().occlusion);
        assert!(tools.pending.is_none());
    }

    #[test]
    fn cancelling_does_not_enable_the_feature_even_if_the_download_finishes() {
        let mut tools = ready_download();
        tools.start(&Context::default(), Job::Install);
        tools.cancelling = true;
        assert!(finish(&mut tools).is_none());
        assert!(tools.pending.is_none());
    }
}
