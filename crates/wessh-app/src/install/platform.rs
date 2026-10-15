//! Host detection: operating system, GPUs, ffmpeg, and the ONNX Runtime flavour that fits them.

use anyhow::{Result, bail};
use std::fmt;
use wessh_core::Device;
use wessh_core::runtime::locate::Plugin;

/// Oldest NVIDIA driver branch that runs CUDA 13 user-space libraries (R580).
const CUDA13_MIN_DRIVER: u32 = 580;
/// Oldest NVIDIA driver branch with CUDA 12 minor-version compatibility (R525).
const CUDA12_MIN_DRIVER: u32 = 525;
/// TensorRT for RTX runs on Ampere (compute capability 8.0) and newer GeForce / RTX GPUs.
const RTX_MIN_COMPUTE: (u32, u32) = (8, 0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Linux,
    Windows,
    Mac,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

/// The machine wessh is being installed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub os: Os,
    pub arch: Arch,
}

impl Target {
    pub fn host() -> Result<Self> {
        let os = match std::env::consts::OS {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::Mac,
            other => bail!("unsupported operating system: {other}"),
        };
        let arch = match std::env::consts::ARCH {
            "x86_64" => Arch::X86_64,
            "aarch64" => Arch::Aarch64,
            other => bail!("unsupported CPU architecture: {other}"),
        };
        Ok(Self { os, arch })
    }
}

/// Which ONNX Runtime build (and which bundled CUDA libraries) to install.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Flavour {
    /// CPU only; works everywhere.
    Cpu,
    /// NVIDIA, CUDA 12 (driver 525+).
    Cuda12,
    /// NVIDIA, CUDA 13 (driver 580+).
    Cuda13,
    /// Apple Silicon build, which also contains the CoreML provider.
    Coreml,
    /// NVIDIA RTX (Ampere+, driver 580+): the CPU build plus NVIDIA's TensorRT for RTX plugin,
    /// which needs only the driver — no CUDA toolkit, cuDNN or cuBLAS.
    Rtx,
    /// AMD / Intel GPUs: the CPU build plus the WebGPU plugin (Vulkan on Linux, D3D12 on Windows).
    Webgpu,
}

impl Flavour {
    /// Major CUDA version whose libraries this flavour needs.
    #[must_use]
    pub const fn cuda_major(self) -> Option<u32> {
        match self {
            Self::Cuda12 => Some(12),
            Self::Cuda13 => Some(13),
            Self::Cpu | Self::Coreml | Self::Rtx | Self::Webgpu => None,
        }
    }

    /// The execution-provider plugin this flavour adds to the CPU build of ONNX Runtime.
    #[must_use]
    pub const fn plugin(self) -> Option<Plugin> {
        match self {
            Self::Rtx => Some(Plugin::NvRtx),
            Self::Webgpu => Some(Plugin::WebGpu),
            Self::Cpu | Self::Cuda12 | Self::Cuda13 | Self::Coreml => None,
        }
    }
}

impl fmt::Display for Flavour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "CPU",
            Self::Cuda12 => "CUDA 12",
            Self::Cuda13 => "CUDA 13",
            Self::Coreml => "CPU + CoreML",
            Self::Rtx => "TensorRT for RTX",
            Self::Webgpu => "WebGPU",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl Vendor {
    fn from_pci_id(id: &str) -> Self {
        match id.trim().to_ascii_lowercase().as_str() {
            "0x10de" => Self::Nvidia,
            "0x1002" => Self::Amd,
            "0x8086" => Self::Intel,
            _ => Self::Other,
        }
    }

    fn from_name(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        if name.contains("nvidia") {
            Self::Nvidia
        } else if name.contains("amd") || name.contains("radeon") {
            Self::Amd
        } else if name.contains("intel") {
            Self::Intel
        } else if name.contains("apple") {
            Self::Apple
        } else {
            Self::Other
        }
    }
}

/// A graphics adapter found on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpu {
    pub vendor: Vendor,
    pub name: String,
    /// Driver version, known for NVIDIA GPUs.
    pub driver: Option<String>,
    /// CUDA compute capability (`(8, 9)` for Ada), known for NVIDIA GPUs.
    pub compute: Option<(u32, u32)>,
}

impl Gpu {
    /// Driver branch, e.g. `615` for `615.71.09`.
    #[must_use]
    pub fn driver_major(&self) -> Option<u32> {
        self.driver.as_deref()?.split('.').next()?.parse().ok()
    }

    /// Whether TensorRT for RTX supports this GPU.
    #[must_use]
    pub fn is_rtx_capable(&self) -> bool {
        self.vendor == Vendor::Nvidia && self.compute.is_some_and(|c| c >= RTX_MIN_COMPUTE)
    }
}

impl fmt::Display for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.driver {
            Some(driver) => write!(f, "{} (driver {driver})", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// Every GPU we can identify without extra dependencies.
#[must_use]
pub fn detect_gpus(target: Target) -> Vec<Gpu> {
    // Drivers older than R510 reject the `compute_cap` field (and with it the whole query).
    let mut gpus = run_stdout("nvidia-smi", &["--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
        .map(|out| parse_nvidia_smi(&out, true))
        .or_else(|| {
            run_stdout("nvidia-smi", &["--query-gpu=name,driver_version", "--format=csv,noheader"]).map(|out| parse_nvidia_smi(&out, false))
        })
        .unwrap_or_default();
    let has_nvidia = !gpus.is_empty();
    match target.os {
        Os::Linux => {
            if !has_nvidia
                && let Some(driver) =
                    std::fs::read_to_string("/proc/driver/nvidia/version").ok().and_then(|v| parse_proc_driver_version(&v))
            {
                gpus.push(Gpu {
                    vendor: Vendor::Nvidia,
                    name: "NVIDIA GPU (nvidia-smi not found)".to_owned(),
                    driver: Some(driver),
                    compute: None,
                });
            }
            let known_nvidia = !gpus.is_empty();
            gpus.extend(linux_drm_gpus().filter(|g| !(known_nvidia && g.vendor == Vendor::Nvidia)));
        }
        Os::Windows => {
            let names =
                run_stdout("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]).unwrap_or_default();
            gpus.extend(
                names
                    .lines()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(|name| Gpu { vendor: Vendor::from_name(name), name: name.to_owned(), driver: None, compute: None })
                    .filter(|g| !(has_nvidia && g.vendor == Vendor::Nvidia)),
            );
        }
        Os::Mac => {
            if target.arch == Arch::Aarch64 {
                let name = run_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
                    .map_or_else(|| "Apple Silicon".to_owned(), |s| s.trim().to_owned());
                gpus.push(Gpu { vendor: Vendor::Apple, name, driver: None, compute: None });
            }
        }
    }
    gpus
}

/// GPUs registered with the kernel's DRM subsystem (`/sys/class/drm/cardN`).
fn linux_drm_gpus() -> impl Iterator<Item = Gpu> {
    let cards = std::fs::read_dir("/sys/class/drm").map(|dir| dir.flatten().map(|e| e.path()).collect::<Vec<_>>()).unwrap_or_default();
    cards
        .into_iter()
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().strip_prefix("card").is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit())))
        })
        .filter_map(|card| {
            let vendor = Vendor::from_pci_id(&std::fs::read_to_string(card.join("device/vendor")).ok()?);
            let name = match vendor {
                Vendor::Nvidia => "NVIDIA GPU",
                Vendor::Amd => "AMD GPU",
                Vendor::Intel => "Intel GPU",
                Vendor::Apple | Vendor::Other => "GPU",
            };
            Some(Gpu { vendor, name: name.to_owned(), driver: None, compute: None })
        })
}

/// `nvidia-smi --query-gpu=name,driver_version[,compute_cap] --format=csv,noheader` output.
/// Names may contain commas, so fields are taken from the right.
fn parse_nvidia_smi(out: &str, with_compute: bool) -> Vec<Gpu> {
    out.lines()
        .filter_map(|line| {
            let (line, compute) = if with_compute {
                let (rest, compute) = line.rsplit_once(',')?;
                (rest, parse_compute(compute))
            } else {
                (line, None)
            };
            let (name, driver) = line.rsplit_once(',')?;
            Some(Gpu { vendor: Vendor::Nvidia, name: name.trim().to_owned(), driver: Some(driver.trim().to_owned()), compute })
        })
        .collect()
}

/// `8.9` → `(8, 9)`.
fn parse_compute(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Driver version from `/proc/driver/nvidia/version` ("NVRM version: ... Kernel Module  615.71.09  Release Build ...").
fn parse_proc_driver_version(text: &str) -> Option<String> {
    let line = text.lines().find(|l| l.starts_with("NVRM"))?;
    line.split_whitespace().find(|tok| tok.contains('.') && tok.chars().all(|c| c.is_ascii_digit() || c == '.')).map(str::to_owned)
}

/// The recommended flavour and a one-line explanation of why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub flavour: Flavour,
    pub reason: String,
}

/// Pick the fastest runtime flavour that will actually work on this machine.
#[must_use]
pub fn choose_flavour(target: Target, gpus: &[Gpu]) -> Choice {
    let choice = |flavour, reason: String| Choice { flavour, reason };
    if target.os == Os::Mac {
        return match target.arch {
            Arch::Aarch64 => choice(Flavour::Coreml, "Apple Silicon".to_owned()),
            Arch::X86_64 => choice(Flavour::Cpu, "Intel Mac".to_owned()),
        };
    }
    let Some(nvidia) = gpus.iter().find(|g| g.vendor == Vendor::Nvidia) else {
        return match gpus.iter().find(|g| matches!(g.vendor, Vendor::Amd | Vendor::Intel)) {
            Some(gpu) if target.os == Os::Windows || has_vulkan_loader() => {
                choice(Flavour::Webgpu, format!("{} GPU, through WebGPU", if gpu.vendor == Vendor::Amd { "AMD" } else { "Intel" }))
            }
            Some(_) => choice(Flavour::Cpu, "AMD / Intel GPU, but no Vulkan loader (libvulkan.so.1) for WebGPU".to_owned()),
            None => choice(Flavour::Cpu, "no GPU found".to_owned()),
        };
    };
    // TensorRT for RTX: about a tenth of the CUDA download and faster (wessh bench, RTX 4050:
    // inswapper 21.5 vs 33.0 ms a frame, p95 23.8 vs 43.6 ms; GFPGAN 126 vs 193 ms).
    if nvidia.is_rtx_capable() && nvidia.driver_major().is_some_and(|major| major >= CUDA13_MIN_DRIVER) {
        return choice(Flavour::Rtx, format!("{} with driver {}", nvidia.name, nvidia.driver.as_deref().unwrap_or_default()));
    }
    cuda_choice(target, nvidia)
}

/// The flavour for an explicit `--device`; `Auto` is [`choose_flavour`].
#[must_use]
pub fn choose_for_device(target: Target, gpus: &[Gpu], device: Device) -> Choice {
    let explicit = |flavour| Choice { flavour, reason: format!("--device {}", format!("{device:?}").to_lowercase()) };
    match device {
        Device::Auto => choose_flavour(target, gpus),
        Device::Rtx => explicit(Flavour::Rtx),
        Device::Webgpu => explicit(Flavour::Webgpu),
        Device::Coreml => explicit(Flavour::Coreml),
        Device::Cpu => explicit(Flavour::Cpu),
        Device::Cuda | Device::Tensorrt => match gpus.iter().find(|g| g.vendor == Vendor::Nvidia) {
            Some(nvidia) => cuda_choice(target, nvidia),
            None => Choice { flavour: Flavour::Cpu, reason: "no NVIDIA GPU found".to_owned() },
        },
    }
}

/// The CUDA build for an NVIDIA GPU's driver.
fn cuda_choice(target: Target, nvidia: &Gpu) -> Choice {
    let choice = |flavour, reason: String| Choice { flavour, reason };
    if target.arch != Arch::X86_64 {
        return choice(Flavour::Cpu, "ONNX Runtime publishes CUDA builds for x86-64 only".to_owned());
    }
    match nvidia.driver_major() {
        Some(major) if major >= CUDA13_MIN_DRIVER => choice(Flavour::Cuda13, format!("NVIDIA driver {major} >= {CUDA13_MIN_DRIVER}")),
        Some(major) if major >= CUDA12_MIN_DRIVER => {
            choice(Flavour::Cuda12, format!("NVIDIA driver {major} >= {CUDA12_MIN_DRIVER} (update to {CUDA13_MIN_DRIVER}+ for CUDA 13)"))
        }
        Some(major) => choice(Flavour::Cpu, format!("NVIDIA driver {major} is older than {CUDA12_MIN_DRIVER}; update it to use the GPU")),
        None => choice(Flavour::Cpu, "NVIDIA GPU found but its driver version is unknown (is the driver installed?)".to_owned()),
    }
}

/// Whether the Vulkan loader the WebGPU plugin needs on Linux is installed.
fn has_vulkan_loader() -> bool {
    ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/lib/x86_64-linux-gnu", "/lib64"]
        .iter()
        .any(|dir| std::path::Path::new(dir).join("libvulkan.so.1").exists())
}

/// First line of `ffmpeg -version` for the ffmpeg wessh would run, or `None` when there is none.
#[must_use]
pub fn ffmpeg_version() -> Option<String> {
    let program = wessh_media::find_ffmpeg()?;
    run_stdout(&program.to_string_lossy(), &["-hide_banner", "-version"]).and_then(|out| out.lines().next().map(str::to_owned))
}

/// Stdout of a successful command, `None` if it is missing or fails.
pub fn run_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = wessh_core::process::background_command(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINUX_X64: Target = Target { os: Os::Linux, arch: Arch::X86_64 };

    fn nvidia(driver: &str) -> Gpu {
        Gpu { vendor: Vendor::Nvidia, name: "GTX".to_owned(), driver: Some(driver.to_owned()), compute: Some((7, 5)) }
    }

    #[test]
    fn parses_nvidia_smi() {
        let gpus = parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 615.71.09, 8.9\nTesla T4, 581.15, 7.5\n", true);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(gpus[0].driver_major(), Some(615));
        assert_eq!(gpus[0].compute, Some((8, 9)));
        assert!(gpus[0].is_rtx_capable());
        assert_eq!(gpus[1].driver_major(), Some(581));
        assert!(!gpus[1].is_rtx_capable());
        let old = parse_nvidia_smi("Quadro K620, 470.2\n", false);
        assert_eq!((old[0].name.as_str(), old[0].driver_major(), old[0].compute), ("Quadro K620", Some(470), None));
    }

    #[test]
    fn parses_proc_driver_version() {
        let text =
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (dvs-builder@U22)\nGCC version: gcc 15\n";
        assert_eq!(parse_proc_driver_version(text).as_deref(), Some("615.71.09"));
        assert_eq!(parse_proc_driver_version("garbage"), None);
    }

    #[test]
    fn picks_tensorrt_for_rtx_on_ampere_and_newer() {
        let rtx = |driver: &str, compute| Gpu { compute: Some(compute), ..nvidia(driver) };
        assert_eq!(choose_flavour(LINUX_X64, &[rtx("615.71.09", (8, 9))]).flavour, Flavour::Rtx);
        assert_eq!(choose_flavour(LINUX_X64, &[rtx("580.1", (8, 0))]).flavour, Flavour::Rtx);
        let arm = Target { os: Os::Linux, arch: Arch::Aarch64 };
        assert_eq!(choose_flavour(arm, &[rtx("615.1", (12, 0))]).flavour, Flavour::Rtx);
        // An older driver or a pre-Ampere GPU stays on CUDA.
        assert_eq!(choose_flavour(LINUX_X64, &[rtx("570.1", (8, 9))]).flavour, Flavour::Cuda12);
        assert_eq!(choose_flavour(LINUX_X64, &[rtx("615.1", (7, 5))]).flavour, Flavour::Cuda13);
    }

    #[test]
    fn explicit_devices_pick_their_flavour() {
        let gpus = [Gpu { compute: Some((8, 9)), ..nvidia("615.1") }];
        assert_eq!(choose_for_device(LINUX_X64, &gpus, Device::Auto).flavour, Flavour::Rtx);
        assert_eq!(choose_for_device(LINUX_X64, &gpus, Device::Cuda).flavour, Flavour::Cuda13);
        assert_eq!(choose_for_device(LINUX_X64, &gpus, Device::Cpu).flavour, Flavour::Cpu);
        assert_eq!(choose_for_device(LINUX_X64, &[], Device::Tensorrt).flavour, Flavour::Cpu);
    }

    #[test]
    fn picks_cuda_by_driver_branch() {
        assert_eq!(choose_flavour(LINUX_X64, &[nvidia("615.71.09")]).flavour, Flavour::Cuda13);
        assert_eq!(choose_flavour(LINUX_X64, &[nvidia("580.65.06")]).flavour, Flavour::Cuda13);
        assert_eq!(choose_flavour(LINUX_X64, &[nvidia("570.1")]).flavour, Flavour::Cuda12);
        assert_eq!(choose_flavour(LINUX_X64, &[nvidia("470.2")]).flavour, Flavour::Cpu);
        let windows = Target { os: Os::Windows, arch: Arch::X86_64 };
        assert_eq!(choose_flavour(windows, &[nvidia("581.15")]).flavour, Flavour::Cuda13);
    }

    #[test]
    fn falls_back_to_cpu() {
        assert_eq!(choose_flavour(LINUX_X64, &[]).flavour, Flavour::Cpu);
        let amd = Gpu { vendor: Vendor::Amd, name: "Radeon".to_owned(), driver: None, compute: None };
        let windows = Target { os: Os::Windows, arch: Arch::X86_64 };
        assert_eq!(choose_flavour(windows, &[amd]).flavour, Flavour::Webgpu);
        let arm = Target { os: Os::Linux, arch: Arch::Aarch64 };
        assert_eq!(choose_flavour(arm, &[nvidia("615.1")]).flavour, Flavour::Cpu);
        let unknown_driver = Gpu { driver: None, ..nvidia("0") };
        assert_eq!(choose_flavour(LINUX_X64, &[unknown_driver]).flavour, Flavour::Cpu);
    }

    #[test]
    fn apple_silicon_gets_coreml() {
        let mac = Target { os: Os::Mac, arch: Arch::Aarch64 };
        assert_eq!(choose_flavour(mac, &[]).flavour, Flavour::Coreml);
    }

    #[test]
    fn vendor_detection() {
        assert_eq!(Vendor::from_pci_id("0x10de\n"), Vendor::Nvidia);
        assert_eq!(Vendor::from_pci_id("0x1002"), Vendor::Amd);
        assert_eq!(Vendor::from_name("AMD Radeon RX 7800 XT"), Vendor::Amd);
        assert_eq!(Vendor::from_name("Intel(R) UHD Graphics"), Vendor::Intel);
    }
}
