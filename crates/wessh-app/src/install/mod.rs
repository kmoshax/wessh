//! Making a machine ready to run wessh: the ONNX Runtime build for its GPU and driver, NVIDIA's
//! CUDA / cuDNN libraries from their Python wheels, the curated models
//! ([`wessh_core::assets::required`]) and, for video files and phone cameras only, an ffmpeg when
//! the machine has none — installed the first time one is needed, never up front. Used by
//! `wessh setup`, the app's onboarding, the Video screen and the `video` / `cam` commands.
//!
//! 1. [`Plan::check`] looks at the files on disk only (no network) to find what is missing —
//!    cheap enough to run on every start-up.
//! 2. [`resolve`] asks the package sources for versions, sizes and checksums.
//! 3. [`install`] downloads (resuming earlier partial downloads, also across restarts),
//!    verifies, and extracts.
//!
//! Both report structured [`Event`]s and stop when their cancel flag is set; [`Installer`]
//! runs them on a background thread. Pausing is cancelling: whatever was downloaded stays on
//! disk and the next run picks it up.

pub mod archive;
pub mod fetch;
pub mod platform;
pub mod progress;
pub mod sources;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use fetch::{Cancelled, Download, Transfer};
use platform::{Choice, Flavour, Gpu, Target};
use progress::Rate;
use sources::Wheel;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;
use wessh_core::assets::{self, Asset, OccluderModel, SwapModel, data_dir};
use wessh_core::runtime::locate;

/// What to install.
#[derive(Clone, Debug)]
pub struct Request {
    /// ONNX Runtime release.
    pub ort_version: String,
    /// Reinstall components that are already present.
    pub force: bool,
    /// Models to install.
    pub models: Vec<Asset>,
    /// The runtime comes from elsewhere (`--ort`, `ORT_DYLIB_PATH`, next to the executable),
    /// so neither it nor its CUDA libraries are installed.
    pub external_runtime: bool,
    /// Install an ffmpeg if there is none (video files and network cameras need one; live
    /// webcam mode doesn't).
    pub ffmpeg: bool,
}

impl Request {
    /// Only an ffmpeg, for the Video screen.
    #[must_use]
    pub fn ffmpeg_only() -> Self {
        Self { ort_version: String::new(), force: false, models: Vec::new(), external_runtime: true, ffmpeg: true }
    }
}

/// The machine being set up.
#[derive(Clone, Debug)]
pub struct Host {
    pub target: Target,
    pub gpus: Vec<Gpu>,
    /// The runtime build and why.
    pub choice: Choice,
}

impl Host {
    /// Detect the OS, GPUs and driver; `flavour` overrides the runtime build that fits them.
    pub fn detect(flavour: Option<Flavour>) -> Result<Self> {
        let target = Target::host()?;
        let gpus = platform::detect_gpus(target);
        let choice = match flavour {
            Some(flavour) => Choice { flavour, reason: "chosen explicitly".to_owned() },
            None => platform::choose_flavour(target, &gpus),
        };
        Ok(Self { target, gpus, choice })
    }

    /// Detect the machine and pick the runtime build `device` asks for.
    pub fn for_device(device: wessh_core::Device) -> Result<Self> {
        let target = Target::host()?;
        let gpus = platform::detect_gpus(target);
        let choice = platform::choose_for_device(target, &gpus, device);
        Ok(Self { target, gpus, choice })
    }

    /// The GPU the runtime will use, for display.
    #[must_use]
    pub fn gpu_name(&self) -> Option<&str> {
        let vendors: &[platform::Vendor] = match self.choice.flavour {
            Flavour::Cuda12 | Flavour::Cuda13 | Flavour::Rtx => &[platform::Vendor::Nvidia],
            Flavour::Webgpu => &[platform::Vendor::Amd, platform::Vendor::Intel],
            Flavour::Coreml => &[platform::Vendor::Apple],
            Flavour::Cpu => return None,
        };
        self.gpus.iter().find(|g| vendors.contains(&g.vendor)).map(|g| g.name.as_str())
    }
}

/// What kind of component an item is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Runtime,
    CudaLibrary,
    /// An execution-provider plugin (with the libraries it brings).
    Plugin,
    Model,
    Ffmpeg,
}

/// Where a component ends up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Destination {
    /// Shared libraries extracted from an archive into this directory.
    Libraries(PathBuf),
    /// A single file downloaded to this path.
    File(PathBuf),
    /// One executable extracted from an archive to this path.
    Program(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Source {
    Runtime(Flavour),
    Wheel(Wheel),
    Model(Asset),
    Ffmpeg,
}

/// One component of the installation.
#[derive(Clone, Debug)]
pub struct Item {
    /// Package or file name.
    pub name: String,
    /// What people call it.
    pub label: String,
    pub kind: Kind,
    /// Version to install, once resolved (empty for models).
    pub version: String,
    pub destination: Destination,
    source: Source,
    pub installed: bool,
    /// Where to get it; `None` until resolved (and for installed items).
    pub download: Option<Download>,
    /// Bytes already downloaded by an earlier, interrupted run.
    pub partial: u64,
}

impl Item {
    /// Download size, when known.
    #[must_use]
    pub fn size(&self) -> Option<u64> {
        self.download.as_ref().and_then(|d| d.size)
    }

    /// Where the download is saved before it is installed.
    fn download_path(&self, download: &Download) -> PathBuf {
        match &self.destination {
            Destination::File(path) => path.clone(),
            Destination::Libraries(_) | Destination::Program(_) => downloads_dir().join(&download.file_name),
        }
    }
}

/// Everything that makes up an installation, and what of it is already there.
#[derive(Clone, Debug)]
pub struct Plan {
    pub host: Host,
    pub items: Vec<Item>,
    ort_version: String,
}

impl Plan {
    /// What is installed and what isn't, judged from the files on disk alone.
    #[must_use]
    pub fn check(host: Host, request: &Request) -> Self {
        let pending = |ready: bool| request.force || !ready;
        let flavour = host.choice.flavour;
        let runtime_dir = locate::default_runtime_dir();
        let mut items = Vec::new();
        if !request.external_runtime {
            // A CUDA build left from an earlier flavour has the core library but the wrong providers.
            let ready = runtime_dir.join(locate::RUNTIME_LIB).exists()
                && flavour.cuda_major().is_some() == locate::has_provider(&runtime_dir, "cuda");
            items.push(Item {
                name: format!("onnxruntime ({flavour})"),
                label: "ONNX Runtime".to_owned(),
                kind: Kind::Runtime,
                version: String::new(),
                destination: Destination::Libraries(runtime_dir.clone()),
                source: Source::Runtime(flavour),
                installed: !pending(ready),
                download: None,
                partial: 0,
            });
            if let Some(cuda_major) = flavour.cuda_major() {
                let cuda_dir = runtime_dir.join("cuda");
                for wheel in sources::cuda_wheels_for_runtime(cuda_major, &request.ort_version) {
                    items.push(Item {
                        name: wheel.package.to_owned(),
                        label: wheel.label.to_owned(),
                        kind: Kind::CudaLibrary,
                        version: String::new(),
                        destination: Destination::Libraries(cuda_dir.clone()),
                        installed: !pending(wheel.is_installed(&cuda_dir, host.target.os)),
                        source: Source::Wheel(wheel),
                        download: None,
                        partial: 0,
                    });
                }
            }
            if let Some(plugin) = flavour.plugin() {
                let wheel = sources::plugin_wheel(plugin);
                let dir = locate::plugin_dir(&runtime_dir, plugin);
                items.push(Item {
                    name: wheel.package.to_owned(),
                    label: wheel.label.to_owned(),
                    kind: Kind::Plugin,
                    version: String::new(),
                    installed: !pending(locate::plugin_library(&runtime_dir, plugin).is_some()),
                    destination: Destination::Libraries(dir),
                    source: Source::Wheel(wheel),
                    download: None,
                    partial: 0,
                });
            }
        }
        for asset in &request.models {
            let path = asset.path();
            items.push(Item {
                name: asset.file.to_owned(),
                label: model_label(asset),
                kind: Kind::Model,
                version: String::new(),
                installed: !pending(path.exists()),
                partial: fetch::file_len(&fetch::partial_path(&path)),
                destination: Destination::File(path),
                source: Source::Model(*asset),
                download: None,
            });
        }
        if request.ffmpeg && wessh_media::system_ffmpeg().is_none() {
            let path = wessh_media::bundled_ffmpeg();
            items.push(Item {
                name: sources::FFMPEG_PACKAGE.to_owned(),
                label: "Video tools (ffmpeg)".to_owned(),
                kind: Kind::Ffmpeg,
                version: String::new(),
                installed: !pending(path.is_file()),
                destination: Destination::Program(path),
                source: Source::Ffmpeg,
                download: None,
                partial: 0,
            });
        }
        Self { host, items, ort_version: request.ort_version.clone() }
    }

    /// Nothing left to install.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.items.iter().all(|i| i.installed)
    }

    /// Indices of the items still to install.
    pub fn pending(&self) -> impl Iterator<Item = usize> + '_ {
        self.items.iter().enumerate().filter(|(_, i)| !i.installed).map(|(index, _)| index)
    }

    /// Total download size of the items still to install, where known.
    #[must_use]
    pub fn download_bytes(&self, kind: Option<Kind>) -> u64 {
        self.pending().filter(|&i| kind.is_none_or(|k| self.items[i].kind == k)).filter_map(|i| self.items[i].size()).sum()
    }

    /// An earlier run left partial downloads behind.
    #[must_use]
    pub fn has_partial_downloads(&self) -> bool {
        self.pending().any(|i| self.items[i].partial > 0)
    }
}

/// Friendly name of a curated model.
pub(crate) fn model_label(asset: &Asset) -> String {
    let role = if *asset == assets::DETECTOR {
        "Face detector"
    } else if *asset == assets::RECOGNIZER {
        "Face recognizer"
    } else if *asset == assets::ENHANCER {
        "Face enhancer"
    } else if *asset == wessh_core::face::masker::PARSER_ASSET {
        "Face parser"
    } else if let Some(model) = SwapModel::ALL.into_iter().find(|m| m.asset() == *asset) {
        return format!("Face swap · {}", model.label());
    } else if let Some(model) = OccluderModel::ALL.into_iter().find(|m| m.asset() == *asset) {
        return format!("Occlusion · {}", model.label());
    } else {
        return asset.name.to_owned();
    };
    role.to_owned()
}

/// Where archives wait to be extracted (and partial downloads of them survive restarts).
fn downloads_dir() -> PathBuf {
    data_dir().join("downloads")
}

/// What is happening to one item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Queued,
    /// Looking up its version, size and checksum.
    Resolving,
    Downloading,
    Verifying,
    Extracting,
    Done,
    Failed(String),
}

/// Progress report from [`resolve`] and [`install`].
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Stage {
        item: usize,
        stage: Stage,
    },
    /// The item's download is known.
    Resolved {
        item: usize,
        version: String,
        size: Option<u64>,
        partial: u64,
    },
    /// `done` bytes of the item are on disk; the rate is averaged over the last few seconds.
    Bytes {
        item: usize,
        done: u64,
        bytes_per_sec: Option<f64>,
    },
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done,
    Cancelled,
    Failed(String),
}

/// Where a run sends its events and how it learns it should stop.
pub struct Reporter {
    events: Sender<Message>,
    cancel: Arc<AtomicBool>,
    notify: Box<dyn Fn() + Send>,
}

impl Reporter {
    fn send(&self, event: Event) {
        let _ = self.events.send(Message::Event(event));
        (self.notify)();
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) { Err(Cancelled.into()) } else { Ok(()) }
    }
}

/// Look up the download of every item still to install.
pub fn resolve(plan: &mut Plan, reporter: &Reporter) -> Result<()> {
    let target = plan.host.target;
    for index in plan.pending().collect::<Vec<_>>() {
        let item = &plan.items[index];
        if item.download.is_some() {
            continue;
        }
        reporter.check_cancel()?;
        reporter.send(Event::Stage { item: index, stage: Stage::Resolving });
        let resolved = match &item.source {
            Source::Runtime(flavour) => sources::ort_release(target, *flavour, &plan.ort_version).map(|d| (plan.ort_version.clone(), d)),
            Source::Wheel(wheel) => wheel.resolve(target),
            Source::Model(asset) => sources::model_download(asset).map(|d| (String::new(), d)),
            Source::Ffmpeg => sources::ffmpeg_download(target),
        };
        let (version, download) = match resolved {
            Ok(resolved) => resolved,
            Err(e) => {
                let error = e.context(format!("could not look up {}", item.label));
                reporter.send(Event::Stage { item: index, stage: Stage::Failed(format!("{error:#}")) });
                return Err(error);
            }
        };
        let item = &mut plan.items[index];
        item.partial = fetch::file_len(&fetch::partial_path(&item.download_path(&download)));
        reporter.send(Event::Resolved { item: index, version: version.clone(), size: download.size, partial: item.partial });
        reporter.send(Event::Stage { item: index, stage: Stage::Queued });
        item.version = version;
        item.download = Some(download);
    }
    Ok(())
}

/// Download and install every item still to install (resolving first where needed).
pub fn install(plan: &mut Plan, reporter: &Reporter) -> Result<()> {
    resolve(plan, reporter)?;
    for index in plan.pending().collect::<Vec<_>>() {
        reporter.check_cancel()?;
        if let Err(e) = install_item(&mut plan.items[index], index, reporter) {
            if !e.is::<Cancelled>() {
                reporter.send(Event::Stage { item: index, stage: Stage::Failed(format!("{e:#}")) });
            }
            return Err(e);
        }
    }
    // Only leftovers of interrupted downloads could remain; an empty directory is noise.
    let _ = std::fs::remove_dir(downloads_dir());
    Ok(())
}

fn install_item(item: &mut Item, index: usize, reporter: &Reporter) -> Result<()> {
    let download = item.download.clone().context("not resolved")?;
    let path = item.download_path(&download);
    reporter.send(Event::Stage { item: index, stage: Stage::Downloading });
    let mut rate = Rate::default();
    let mut on_progress = |transfer| match transfer {
        Transfer::Bytes(done) => {
            rate.push(Instant::now(), done);
            item.partial = done;
            reporter.send(Event::Bytes { item: index, done, bytes_per_sec: rate.bytes_per_sec() });
        }
        Transfer::Verifying => reporter.send(Event::Stage { item: index, stage: Stage::Verifying }),
    };
    fetch::download(&download, &path, &reporter.cancel, &mut on_progress)?;
    match &item.destination {
        Destination::Libraries(dir) => {
            reporter.send(Event::Stage { item: index, stage: Stage::Extracting });
            archive::extract_libraries(&path, dir, &reporter.cancel)?;
            std::fs::remove_file(&path)?;
        }
        Destination::Program(program) => {
            reporter.send(Event::Stage { item: index, stage: Stage::Extracting });
            archive::extract_program(&path, sources::FFMPEG_MEMBER_PREFIX, program)?;
            std::fs::remove_file(&path)?;
        }
        Destination::File(_) => {}
    }
    item.installed = true;
    reporter.send(Event::Stage { item: index, stage: Stage::Done });
    Ok(())
}

/// What an [`Installer`] run does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Job {
    /// Only look up downloads and sizes.
    Resolve,
    /// Resolve what isn't yet, then download and install.
    Install,
}

/// What an [`Installer`] reports.
#[derive(Debug)]
pub enum Message {
    Event(Event),
    /// The run ended; the plan comes back with everything it learnt and installed.
    Finished {
        plan: Box<Plan>,
        outcome: Outcome,
    },
}

/// A [`resolve`] or [`install`] run on a background thread.
pub struct Installer {
    messages: Receiver<Message>,
    cancel: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Installer {
    /// Stop the run and wait for it, so no curl process outlives the app (a second one would
    /// later append to the same partial file). Every stage checks the flag within moments.
    fn drop(&mut self) {
        self.cancel();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Installer {
    /// Start `job` on `plan`; `notify` is called after every message (e.g. to repaint).
    pub fn spawn(mut plan: Plan, job: Job, notify: impl Fn() + Send + 'static) -> Self {
        let (events, messages) = crossbeam_channel::unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let reporter = Reporter { events, cancel: cancel.clone(), notify: Box::new(notify) };
        let thread = std::thread::Builder::new()
            .name("install".into())
            .spawn(move || {
                let result = match job {
                    Job::Resolve => resolve(&mut plan, &reporter),
                    Job::Install => install(&mut plan, &reporter),
                };
                let outcome = match result {
                    Ok(()) => Outcome::Done,
                    Err(e) if e.is::<Cancelled>() => Outcome::Cancelled,
                    Err(e) => Outcome::Failed(format!("{e:#}")),
                };
                let _ = reporter.events.send(Message::Finished { plan: Box::new(plan), outcome });
                (reporter.notify)();
            })
            .expect("spawning the install thread");
        Self { messages, cancel, thread: Some(thread) }
    }

    /// Stop as soon as possible; partial downloads are kept. A [`Message::Finished`] with
    /// [`Outcome::Cancelled`] follows.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Messages that arrived since the last call.
    pub fn poll(&self) -> impl Iterator<Item = Message> + '_ {
        self.messages.try_iter()
    }

    /// Wait for the next message; `None` once the run has ended and everything was read.
    pub fn next(&self) -> Option<Message> {
        self.messages.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::{Arch, Os};
    use wessh_core::Settings;

    fn host(flavour: Flavour) -> Host {
        Host { target: Target { os: Os::Linux, arch: Arch::X86_64 }, gpus: Vec::new(), choice: Choice { flavour, reason: String::new() } }
    }

    fn request() -> Request {
        Request {
            ort_version: sources::ORT_VERSION.to_owned(),
            force: true,
            models: assets::required(&Settings::default()),
            external_runtime: false,
            ffmpeg: false,
        }
    }

    #[test]
    fn cuda_plan_has_runtime_libraries_and_models() {
        let plan = Plan::check(host(Flavour::Cuda13), &request());
        let count = |kind| plan.items.iter().filter(|i| i.kind == kind).count();
        assert_eq!(count(Kind::Runtime), 1);
        assert_eq!(count(Kind::CudaLibrary), sources::cuda_wheels_for_runtime(13, sources::ORT_VERSION).len());
        assert_eq!(count(Kind::Model), assets::required(&Settings::default()).len());
        assert!(!plan.is_complete());
        assert!(plan.items.iter().all(|i| i.download.is_none()));
    }

    #[test]
    fn cuda_face_setup_does_not_schedule_fft_download() {
        let plan = Plan::check(host(Flavour::Cuda13), &request());
        assert_eq!(plan.items.iter().filter(|item| item.kind == Kind::CudaLibrary).count(), 6);
        assert!(!plan.items.iter().any(|item| item.label == "cuFFT"));
    }

    #[test]
    fn plugin_flavours_add_their_plugin_to_the_cpu_runtime() {
        for (flavour, plugin) in [(Flavour::Rtx, locate::Plugin::NvRtx), (Flavour::Webgpu, locate::Plugin::WebGpu)] {
            let plan = Plan::check(host(flavour), &request());
            let kinds: Vec<Kind> = plan.items.iter().map(|i| i.kind).filter(|&k| k != Kind::Model).collect();
            assert_eq!(kinds, [Kind::Runtime, Kind::Plugin]);
            let item = plan.items.iter().find(|i| i.kind == Kind::Plugin).unwrap();
            assert_eq!(item.destination, Destination::Libraries(locate::plugin_dir(&locate::default_runtime_dir(), plugin)));
        }
    }

    #[test]
    fn cpu_plan_has_no_gpu_libraries() {
        let plan = Plan::check(host(Flavour::Cpu), &request());
        assert!(plan.items.iter().all(|i| matches!(i.kind, Kind::Runtime | Kind::Model)));
        let external = Request { external_runtime: true, ..request() };
        assert!(Plan::check(host(Flavour::Cuda13), &external).items.iter().all(|i| i.kind == Kind::Model));
    }

    #[test]
    fn ffmpeg_only_installs_ffmpeg_when_the_machine_has_none() {
        let plan = Plan::check(host(Flavour::Cuda13), &Request::ffmpeg_only());
        let expected = usize::from(wessh_media::system_ffmpeg().is_none());
        assert_eq!(plan.items.len(), expected);
        assert!(plan.items.iter().all(|i| i.kind == Kind::Ffmpeg && matches!(i.destination, Destination::Program(_))));
    }

    #[test]
    fn curated_models_have_friendly_labels() {
        let labels: Vec<String> = assets::required(&Settings::default()).iter().map(model_label).collect();
        assert!(labels.iter().all(|label| !label.contains('_')), "{labels:?}");
        assert!(labels.iter().any(|label| label.starts_with("Face swap")));
    }

    #[test]
    fn sizes_count_only_what_is_left() {
        let mut plan = Plan::check(host(Flavour::Cpu), &request());
        let download = |size| Download { url: String::new(), file_name: "x".to_owned(), size: Some(size), sha256: None };
        plan.items[0].download = Some(download(100));
        plan.items[0].partial = 40;
        plan.items[1].download = Some(download(10));
        plan.items[1].installed = true;
        assert_eq!(plan.download_bytes(None), 100);
        assert_eq!(plan.download_bytes(Some(Kind::Model)), 0);
        assert!(plan.has_partial_downloads());
    }
}
