//! Camera and virtual-camera discovery.
//!
//! - Linux: `/sys/class/video4linux`, no device is opened. A UVC webcam exposes a capture node
//!   (`index` 0) and a metadata node (`index` 1); v4l2loopback devices are virtual.
//! - Windows: `ffmpeg -list_devices true -f dshow`; macOS: `ffmpeg -list_devices true -f avfoundation`.
//!
//! Parsing is kept apart from I/O so it is unit-tested on captured listings.

use crate::ffmpeg::ffmpeg_program;
use std::path::Path;
use std::process::Stdio;

/// The command that creates a v4l2loopback device wajh can write to (`/dev/video10`, named "wajh").
pub const LOOPBACK_MODPROBE: &str = "sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=wajh exclusive_caps=1";

const SYSFS_VIDEO: &str = "/sys/class/video4linux";
const SYSFS_LOOPBACK_MODULE: &str = "/sys/module/v4l2loopback";
/// Driver name v4l2loopback reports (as a platform driver on recent versions).
const LOOPBACK_DRIVER: &str = "v4l2loopback";
/// Card label v4l2loopback uses when `card_label` is not given.
const LOOPBACK_DEFAULT_LABEL: &str = "Dummy video device";
/// Every V4L2 driver numbers the nodes of one device from 0; extra nodes carry metadata.
const PRIMARY_NODE_INDEX: u32 = 0;

/// One camera or virtual-camera device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoDevice {
    /// What ffmpeg opens: `/dev/video0`, an avfoundation index, or `video=<name>` for dshow.
    pub id: String,
    /// Human-readable name.
    pub name: String,
}

/// Whether frames can be sent to a v4l2loopback virtual camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopbackStatus {
    /// Not Linux: virtual-camera output goes over a network stream instead.
    #[default]
    Unsupported,
    /// Linux, but the `v4l2loopback` kernel module is not loaded.
    NotLoaded,
    /// The module is loaded (devices are listed in [`Devices::outputs`]).
    Loaded,
}

/// Everything discovered in one scan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Devices {
    pub cameras: Vec<VideoDevice>,
    /// v4l2loopback devices (Linux only).
    pub outputs: Vec<VideoDevice>,
    pub loopback: LoopbackStatus,
}

/// Scan for cameras and virtual-camera outputs. Blocking (runs ffmpeg on Windows and macOS).
#[must_use]
pub fn list_devices() -> Devices {
    match std::env::consts::OS {
        "linux" => {
            let nodes = read_sysfs(Path::new(SYSFS_VIDEO));
            let loaded = Path::new(SYSFS_LOOPBACK_MODULE).exists();
            classify_v4l2(&nodes, loaded)
        }
        "macos" => Devices { cameras: parse_avfoundation(&list_ffmpeg_devices("avfoundation", "")), ..Devices::default() },
        _ => Devices { cameras: parse_dshow(&list_ffmpeg_devices("dshow", "dummy")), ..Devices::default() },
    }
}

/// What sysfs says about one `/dev/videoN` node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct V4l2Node {
    /// `video0`, …
    pub node: String,
    /// The card name (`name` attribute).
    pub name: String,
    /// Node number within its device (`index` attribute).
    pub index: Option<u32>,
    /// Kernel driver bound to the parent device, e.g. `uvcvideo`.
    pub driver: Option<String>,
    /// The node sits under `/sys/devices/virtual` (no hardware behind it).
    pub is_virtual: bool,
}

impl V4l2Node {
    fn is_loopback(&self) -> bool {
        self.driver.as_deref().is_some_and(|d| d.replace(' ', "") == LOOPBACK_DRIVER)
            || (self.is_virtual && self.driver.is_none())
            || self.name.starts_with(LOOPBACK_DEFAULT_LABEL)
    }

    fn device(&self) -> VideoDevice {
        VideoDevice { id: format!("/dev/{}", self.node), name: clean_card_name(&self.name).to_owned() }
    }

    /// Numeric suffix of `videoN`, for natural ordering.
    fn number(&self) -> u32 {
        self.node.trim_start_matches(|c: char| !c.is_ascii_digit()).parse().unwrap_or(u32::MAX)
    }
}

/// UVC cards are named `<product>: <product again, truncated>`; keep the first part.
fn clean_card_name(name: &str) -> &str {
    match name.split_once(": ") {
        Some((product, repeat)) if product.starts_with(repeat.trim()) => product,
        _ => name,
    }
}

/// Split sysfs nodes into capture devices and loopback outputs.
#[must_use]
pub fn classify_v4l2(nodes: &[V4l2Node], loopback_loaded: bool) -> Devices {
    let mut sorted: Vec<&V4l2Node> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.number());
    let (outputs, hardware): (Vec<&V4l2Node>, Vec<&V4l2Node>) = sorted.into_iter().partition(|n| n.is_loopback());
    let cameras = hardware.into_iter().filter(|n| n.index.is_none_or(|i| i == PRIMARY_NODE_INDEX)).map(V4l2Node::device).collect();
    let outputs: Vec<VideoDevice> = outputs.into_iter().map(V4l2Node::device).collect();
    let loaded = loopback_loaded || !outputs.is_empty();
    Devices { cameras, outputs, loopback: if loaded { LoopbackStatus::Loaded } else { LoopbackStatus::NotLoaded } }
}

fn read_sysfs(root: &Path) -> Vec<V4l2Node> {
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    let read = |path: &Path| std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned());
    entries
        .flatten()
        .map(|entry| {
            let dir = entry.path();
            let driver =
                std::fs::read_link(dir.join("device/driver")).ok().and_then(|p| Some(p.file_name()?.to_string_lossy().into_owned()));
            let is_virtual = std::fs::canonicalize(&dir).is_ok_and(|p| p.components().any(|c| c.as_os_str() == "virtual"));
            V4l2Node {
                node: entry.file_name().to_string_lossy().into_owned(),
                name: read(&dir.join("name")).unwrap_or_default(),
                index: read(&dir.join("index")).and_then(|s| s.parse().ok()),
                driver,
                is_virtual,
            }
        })
        .collect()
}

/// ffmpeg prints device lists to stderr at info level, then fails on the dummy input.
fn list_ffmpeg_devices(format: &str, input: &str) -> String {
    let output = ffmpeg_program()
        .args(["-hide_banner", "-nostdin", "-list_devices", "true", "-f", format, "-i", input])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .output();
    output.map(|o| String::from_utf8_lossy(&o.stderr).into_owned()).unwrap_or_default()
}

/// Strip the `[dshow @ 0x…] ` / `[AVFoundation indev @ 0x…] ` prefix.
fn message(line: &str) -> &str {
    match line.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
        Some((_, text)) => text,
        None => line,
    }
}

/// Parse `ffmpeg -list_devices true -f dshow -i dummy` (both the ≥ 5.0 `"name" (video)` form and
/// the older sectioned form). Returns video devices as `video=<name>`.
#[must_use]
pub fn parse_dshow(listing: &str) -> Vec<VideoDevice> {
    let mut devices = Vec::new();
    let mut in_video_section = false;
    for line in listing.lines().map(message) {
        let text = line.trim();
        if text.starts_with("DirectShow video devices") {
            in_video_section = true;
            continue;
        }
        if text.starts_with("DirectShow audio devices") {
            in_video_section = false;
            continue;
        }
        let Some(rest) = text.strip_prefix('"') else { continue };
        let Some((name, kind)) = rest.split_once('"') else { continue };
        let kind = kind.trim();
        // `(video)`, `(audio, video)`, or nothing in the legacy sectioned form.
        let video = kind.contains("video") || (kind.is_empty() && in_video_section);
        if video && !devices.iter().any(|d: &VideoDevice| d.name == name) {
            devices.push(VideoDevice { id: format!("video={name}"), name: name.to_owned() });
        }
    }
    devices
}

/// Parse `ffmpeg -f avfoundation -list_devices true -i ""`. Returns video devices by index.
#[must_use]
pub fn parse_avfoundation(listing: &str) -> Vec<VideoDevice> {
    let mut devices = Vec::new();
    let mut in_video_section = false;
    for line in listing.lines().map(message) {
        let text = line.trim();
        if text.contains("video devices:") {
            in_video_section = true;
        } else if text.contains("audio devices:") {
            in_video_section = false;
        } else if in_video_section
            && let Some((index, name)) = text.strip_prefix('[').and_then(|rest| rest.split_once("] "))
            && index.parse::<u32>().is_ok()
        {
            devices.push(VideoDevice { id: index.to_owned(), name: name.trim().to_owned() });
        }
    }
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: &str, name: &str, index: Option<u32>, driver: Option<&str>, is_virtual: bool) -> V4l2Node {
        V4l2Node { node: n.into(), name: name.into(), index, driver: driver.map(Into::into), is_virtual }
    }

    #[test]
    fn uvc_metadata_nodes_are_skipped_and_loopback_is_an_output() {
        let nodes = [
            node("video10", "wajh", Some(0), None, true),
            node("video1", "USB Cam", Some(1), Some("uvcvideo"), false),
            node("video0", "USB Cam", Some(0), Some("uvcvideo"), false),
            node("video2", "Dummy video device (0x0000)", Some(0), Some("v4l2 loopback"), false),
        ];
        let devices = classify_v4l2(&nodes, true);
        assert_eq!(devices.cameras, [VideoDevice { id: "/dev/video0".into(), name: "USB Cam".into() }]);
        let outputs: Vec<&str> = devices.outputs.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(outputs, ["/dev/video2", "/dev/video10"]);
        assert_eq!(devices.loopback, LoopbackStatus::Loaded);
    }

    #[test]
    fn uvc_names_lose_their_repeat() {
        assert_eq!(clean_card_name("USB2.0 HD UVC WebCam: USB2.0 HD"), "USB2.0 HD UVC WebCam");
        assert_eq!(clean_card_name("Integrated Camera: Integrated C"), "Integrated Camera");
        assert_eq!(clean_card_name("Cam: Front"), "Cam: Front");
    }

    #[test]
    fn missing_module_is_reported() {
        let devices = classify_v4l2(&[node("video0", "Cam", Some(0), Some("uvcvideo"), false)], false);
        assert_eq!(devices.loopback, LoopbackStatus::NotLoaded);
        assert_eq!(devices.outputs.len(), 0);
    }

    #[test]
    fn parses_modern_dshow_listing() {
        let listing = r#"[dshow @ 0000020b] "Integrated Camera" (video)
[dshow @ 0000020b]   Alternative name "@device_pnp_\\?\usb#vid_04f2"
[dshow @ 0000020b] "OBS Virtual Camera" (none)
[dshow @ 0000020b] "Microphone Array (Realtek)" (audio)
[dshow @ 0000020b] "Logi Cam" (audio, video)
dummy: Immediate exit requested"#;
        let names: Vec<String> = parse_dshow(listing).into_iter().map(|d| d.id).collect();
        assert_eq!(names, ["video=Integrated Camera", "video=Logi Cam"]);
    }

    #[test]
    fn parses_legacy_dshow_listing() {
        let listing = r#"[dshow @ 0x1] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x1]  "HD Webcam"
[dshow @ 0x1]     Alternative name "@device_pnp_x"
[dshow @ 0x1] DirectShow audio devices
[dshow @ 0x1]  "Microphone""#;
        let devices = parse_dshow(listing);
        assert_eq!(devices, [VideoDevice { id: "video=HD Webcam".into(), name: "HD Webcam".into() }]);
    }

    #[test]
    fn parses_avfoundation_listing() {
        let listing = "[AVFoundation indev @ 0x7f] AVFoundation video devices:
[AVFoundation indev @ 0x7f] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f] [1] Capture screen 0
[AVFoundation indev @ 0x7f] AVFoundation audio devices:
[AVFoundation indev @ 0x7f] [0] MacBook Pro Microphone
: Input/output error";
        let devices = parse_avfoundation(listing);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0], VideoDevice { id: "0".into(), name: "FaceTime HD Camera".into() });
    }
}
