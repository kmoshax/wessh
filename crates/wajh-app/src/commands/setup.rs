//! `wajh setup`: one command that makes a machine ready to run wajh.
//!
//! Installs the ONNX Runtime build for the GPU and driver into `<data dir>/ort/lib`, NVIDIA's
//! CUDA / cuDNN / TensorRT libraries into `<data dir>/ort/lib/cuda`, and the curated models into
//! `<data dir>/models` — the same set the app's onboarding installs (see [`crate::install`]).
//! Every step is skipped when already done, downloads resume, and checksums are verified
//! whenever the source publishes one.

use crate::cli::{Options, SetupArgs};
use crate::install::progress::{human_duration, human_rate, human_size, time_left};
use crate::install::{Event, Host, Installer, Job, Message, Outcome, Plan, Request, Stage, platform};
use anyhow::{Result, bail};
use std::io::Write as _;
use wajh_core::assets::{self, data_dir};

pub fn run(args: &SetupArgs, options: &Options) -> Result<()> {
    let host = Host::detect(args.flavour)?;
    let request = Request {
        tensorrt: !args.no_tensorrt,
        ort_version: args.ort_version.clone(),
        force: args.force,
        models: if args.no_models { Vec::new() } else { assets::required(&options.settings()).to_vec() },
        external_runtime: false,
    };

    println!("data dir   {}", data_dir().display());
    if host.gpus.is_empty() {
        println!("GPU        none detected");
    }
    for gpu in &host.gpus {
        println!("GPU        {gpu}");
    }
    println!("runtime    {} ({})", host.choice.flavour, host.choice.reason);
    println!("\nchecking what is needed...");

    let target = host.target;
    let plan = run_job(Plan::check(host, &request), Job::Resolve)?;
    print_plan(&plan);
    let plan = if args.dry_run || plan.is_complete() { plan } else { run_job(plan, Job::Install)? };
    debug_assert!(args.dry_run || plan.is_complete());

    println!();
    match platform::ffmpeg_version() {
        Some(version) => println!("ffmpeg     {version}"),
        None => println!(
            "ffmpeg     not found; video and camera modes need it. Install it with:\n           {}",
            platform::ffmpeg_install_hint(target.os)
        ),
    }
    if args.dry_run {
        println!("\ndry run: nothing was downloaded or changed");
    } else {
        println!("\nready. `wajh doctor` checks the installation; `wajh` opens the app");
    }
    Ok(())
}

/// Run `job` on the installer thread, printing its progress.
fn run_job(plan: Plan, job: Job) -> Result<Plan> {
    let mut printer = Printer {
        names: plan.items.iter().map(|i| i.name.clone()).collect(),
        sizes: plan.items.iter().map(|i| (i.size(), i.partial)).collect(),
        total: plan.pending().count(),
        ..Printer::default()
    };
    let installer = Installer::spawn(plan, job, || {});
    while let Some(message) = installer.next() {
        match message {
            Message::Event(event) => printer.event(event, job),
            Message::Finished { plan, outcome } => {
                printer.end_line();
                return match outcome {
                    Outcome::Done => Ok(*plan),
                    Outcome::Cancelled => bail!("cancelled"),
                    Outcome::Failed(error) => bail!("{error}\nrun the command again to resume"),
                };
            }
        }
    }
    bail!("the install thread stopped unexpectedly")
}

/// Console progress: one line per item, rewritten in place while downloading.
#[derive(Default)]
struct Printer {
    names: Vec<String>,
    /// Items to install.
    total: usize,
    /// Size of each item once resolved, and the bytes kept from an interrupted download.
    sizes: Vec<(Option<u64>, u64)>,
    /// Size of the item being downloaded.
    size: Option<u64>,
    in_line: bool,
    step: usize,
}

impl Printer {
    fn event(&mut self, event: Event, job: Job) {
        match event {
            Event::Resolved { item, size, partial, .. } => self.sizes[item] = (size, partial),
            Event::Stage { item, stage: Stage::Downloading } if job == Job::Install => {
                self.end_line();
                self.step += 1;
                let (size, partial) = self.sizes[item];
                self.size = size;
                println!("\n[{}/{}] {} {}", self.step, self.total, self.names[item], size.map(human_size).unwrap_or_default());
                if partial > 0 {
                    println!("      resuming: {} already downloaded", human_size(partial));
                }
            }
            Event::Stage { stage: Stage::Verifying, .. } => self.status("verifying"),
            Event::Stage { stage: Stage::Extracting, .. } => self.status("extracting"),
            Event::Stage { stage: Stage::Failed(error), .. } => {
                self.end_line();
                eprintln!("      failed: {error}");
            }
            Event::Stage { .. } => {}
            Event::Bytes { done, bytes_per_sec, .. } => {
                let total = self.size.map_or_else(String::new, |size| format!(" / {}", human_size(size)));
                let rate = bytes_per_sec.map_or_else(String::new, |r| format!("  {}", human_rate(r)));
                let left = time_left(self.size.unwrap_or(done).saturating_sub(done), bytes_per_sec)
                    .map_or_else(String::new, |t| format!("  {} left", human_duration(t)));
                self.status(&format!("{}{total}{rate}{left}", human_size(done)));
            }
        }
    }

    fn status(&mut self, text: &str) {
        print!("\r      {text:<60}");
        let _ = std::io::stdout().flush();
        self.in_line = true;
    }

    fn end_line(&mut self) {
        if std::mem::take(&mut self.in_line) {
            println!();
        }
    }
}

fn print_plan(plan: &Plan) {
    let width = plan.items.iter().map(|i| i.name.len()).max().unwrap_or_default();
    for item in &plan.items {
        let (size, action) = if item.installed {
            (String::new(), "installed")
        } else {
            (item.size().map_or_else(|| "size unknown".to_owned(), human_size), "download")
        };
        println!("  {:width$}  {:12}  {size:>10}  {action}", item.name, item.version);
    }
    let pending = plan.pending().count();
    if pending == 0 {
        println!("everything is already installed (use --force to reinstall)");
    } else {
        println!("{pending} download(s), {} in total", human_size(plan.download_bytes(None)));
    }
}
