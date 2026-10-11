//! Where frames and the source face come from, and where results go.

use super::super::{Action, Frame, paint_cover};
use super::{caption, section, section_with, switch_row};
use crate::ui::components::{Button, Chip, CircularProgress, Dropdown, IconButton, SegmentedButton, TextField, paint};
use crate::ui::icons;
use crate::ui::theme::{self, TypeRole, motion, shape, size, spacing};
use eframe::egui::{self, Align2, CursorIcon, Key, Label, Margin, Sense, Stroke, StrokeKind, TextureHandle, Ui, Vec2, pos2, vec2};
use std::path::PathBuf;
use weshy_media::{DEFAULT_LOOPBACK, DEFAULT_STREAM, LOOPBACK_MODPROBE, LoopbackStatus, OutputTarget, VideoDevice};

// ── Source face ────────────────────────────────────────────────────────────────

const THUMBNAIL: f32 = 72.0;
const PENDING_PROGRESS: f32 = 40.0;
const DROP_ZONE_HEIGHT: f32 = 184.0;
const DROP_ICON_CIRCLE: f32 = 56.0;

pub(super) fn source(ui: &mut Ui, frame: &mut Frame<'_>) {
    section(ui, icons::FACE, "Source face", |ui| {
        match frame.source {
            _ if frame.source_pending => source_pending(ui),
            Some(texture) if frame.snapshot.has_source => source_loaded(ui, frame, texture),
            _ => drop_zone(ui, frame),
        }
        if frame.intake.needs_path_field() {
            path_field(ui, frame);
        }
    });
}

/// Spinner shown between choosing a photo and the worker accepting (or rejecting) it.
fn source_pending(ui: &mut Ui) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing::LG;
        ui.add(CircularProgress::new(PENDING_PROGRESS));
        ui.vertical(|ui| {
            ui.label(TypeRole::TitleMedium.rich("Finding the face…", colors.on_surface));
            ui.label(TypeRole::BodyMedium.rich("Detecting and encoding the source photo", colors.on_surface_variant));
        });
    });
}

fn source_loaded(ui: &mut Ui, frame: &mut Frame<'_>, texture: &TextureHandle) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(THUMBNAIL), Sense::hover());
        paint_cover(ui, texture, rect, shape::MEDIUM);
        ui.add_space(spacing::SM);
        ui.vertical(|ui| {
            ui.add_space(spacing::XS);
            let name = frame.snapshot.source_name.as_deref().unwrap_or("Source face");
            ui.add(Label::new(TypeRole::TitleMedium.rich(name, colors.on_surface)).truncate());
            face_ready(ui);
            ui.add_space(spacing::XS);
            if frame.intake.pick && ui.add(Button::tonal("Replace").icon(icons::SWAP_HORIZ)).clicked() {
                frame.actions.push(Action::PickSource);
            }
        });
    });
    if frame.intake.drop && !frame.intake.pick {
        caption(ui, "Drop another photo anywhere to replace it");
    }
}

/// Filled check icon + "Face ready".
fn face_ready(ui: &mut Ui) {
    let colors = theme::colors(ui.ctx());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing::XS;
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON_SMALL), Sense::hover());
        paint::icon(ui.painter(), rect.center(), icons::CHECK_CIRCLE, size::ICON_SMALL, colors.primary, true);
        ui.label(TypeRole::BodyMedium.rich("Face ready", colors.on_surface_variant));
    });
}

/// Empty state: a large outlined target inviting a drop or a click.
fn drop_zone(ui: &mut Ui, frame: &mut Frame<'_>) {
    let colors = theme::colors(ui.ctx());
    let intake = frame.intake;
    let sense = if intake.pick { Sense::click() } else { Sense::hover() };
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), DROP_ZONE_HEIGHT), sense);
    let response = if intake.pick { response.on_hover_cursor(CursorIcon::PointingHand) } else { response };
    let hovered = ui.ctx().animate_bool_with_time(response.id, response.hovered() && intake.pick, motion::SHORT_4);
    let painter = ui.painter();
    let outline = colors.outline_variant.lerp_to_gamma(colors.primary, hovered);
    painter.rect_stroke(rect, shape::MEDIUM, Stroke::new(size::STROKE, outline), StrokeKind::Inside);
    paint::state_layer(ui, &response, rect, shape::MEDIUM, colors.primary);

    let circle_center = pos2(rect.center().x, rect.top() + spacing::XL + DROP_ICON_CIRCLE / 2.0);
    painter.circle_filled(circle_center, DROP_ICON_CIRCLE / 2.0, colors.primary_container);
    paint::icon(painter, circle_center, icons::ADD_PHOTO_ALTERNATE, size::ICON, colors.on_primary_container, false);
    let title_pos = pos2(rect.center().x, circle_center.y + DROP_ICON_CIRCLE / 2.0 + spacing::MD);
    let title = paint::text(ui, title_pos, Align2::CENTER_TOP, TypeRole::TitleMedium, intake.title(), colors.on_surface);
    let hint_pos = pos2(rect.center().x, title.bottom() + spacing::XS);
    paint::text(ui, hint_pos, Align2::CENTER_TOP, TypeRole::BodyMedium, intake.hint(), colors.on_surface_variant);
    if response.clicked() {
        frame.actions.push(Action::PickSource);
    }
}

/// Fallback when there is no file dialog and no drag-and-drop: type a path.
fn path_field(ui: &mut Ui, frame: &mut Frame<'_>) {
    let path = &mut frame.form.source_path;
    let submitted = ui.add(TextField::new("Image path", path).icon(icons::FOLDER_OPEN).hint("Full path to a JPG or PNG")).lost_focus()
        && ui.input(|i| i.key_pressed(Key::Enter));
    let clicked = ui.add_enabled(!path.trim().is_empty(), Button::tonal("Load").icon(icons::UPLOAD)).clicked();
    if submitted || clicked {
        frame.actions.push(Action::LoadSource(PathBuf::from(path.trim())));
    }
}

// ── Devices ────────────────────────────────────────────────────────────────────

/// A dropdown entry: a discovered device or free text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Listed(usize),
    Custom,
}

const CUSTOM_LABEL: &str = "Custom…";

/// A dropdown of `options` (id, label) plus "Custom…", which reveals a text field.
/// `custom` remembers that the user chose to type the value.
fn device_picker(ui: &mut Ui, label: &str, icon: char, options: &[(String, String)], value: &mut String, custom: &mut bool, hint: &str) {
    let entries: Vec<(Choice, &str)> = options
        .iter()
        .enumerate()
        .map(|(i, (_, label))| (Choice::Listed(i), label.as_str()))
        .chain(std::iter::once((Choice::Custom, CUSTOM_LABEL)))
        .collect();
    let listed = options.iter().position(|(id, _)| id == value.trim());
    let mut choice = match listed {
        Some(i) if !*custom => Choice::Listed(i),
        _ => Choice::Custom,
    };
    if ui.add(Dropdown::new(label, &mut choice, &entries).icon(icon)).changed() {
        match choice {
            Choice::Listed(i) => {
                value.clone_from(&options[i].0);
                *custom = false;
            }
            Choice::Custom => *custom = true,
        }
    }
    if choice == Choice::Custom {
        ui.add(TextField::new("Custom", value).hint(hint));
    }
}

/// Dropdown labels: the device name, with its path where names repeat or `with_id` asks for it.
fn device_options(devices: &[VideoDevice], with_id: bool) -> Vec<(String, String)> {
    devices
        .iter()
        .map(|d| {
            let repeated = devices.iter().filter(|other| other.name == d.name).count() > 1;
            let label = if repeated || with_id || d.name.is_empty() { format!("{} · {}", d.name, d.id) } else { d.name.clone() };
            (d.id.clone(), label.trim_start_matches([' ', '·']).trim().to_owned())
        })
        .collect()
}

/// Rescan button for a section header (a spinner while scanning). Returns whether it was clicked.
fn refresh_button(ui: &mut Ui, scanning: bool) -> bool {
    if scanning {
        ui.add_space(spacing::SM);
        ui.add(CircularProgress::new(size::ICON_SMALL));
        false
    } else {
        ui.add(IconButton::new(icons::REFRESH, "Look for devices again")).clicked()
    }
}

// ── Camera ─────────────────────────────────────────────────────────────────────

/// Address format of phone camera apps (IP Webcam's default port and path).
const PHONE_URL_EXAMPLE: &str = "http://192.168.1.20:8080/video";
const PHONE_HELP: &str = "Phone as camera: install IP Webcam (Android) or DroidCam, start its server on the same \
Wi-Fi, and paste the http:// or rtsp:// address it shows.";

const RESOLUTIONS: [((usize, usize), &str); 3] = [((640, 480), "480p"), ((1280, 720), "720p"), ((1920, 1080), "1080p")];
const FRAME_RATES: [(u32, &str); 3] = [(15, "15"), (30, "30"), (60, "60")];

pub(super) fn camera(ui: &mut Ui, frame: &mut Frame<'_>) {
    let live = frame.is_live();
    let options = frame.devices.map(|d| device_options(&d.cameras, false)).unwrap_or_default();
    let scanned = frame.devices.is_some();
    let scanning = frame.scanning;
    let mut refresh = false;
    let form = &mut *frame.form;
    section_with(
        ui,
        icons::VIDEOCAM,
        "Camera",
        |ui| refresh = refresh_button(ui, scanning),
        |ui| {
            ui.add_enabled_ui(!live, |ui| {
                let hint = format!("{} or {PHONE_URL_EXAMPLE}", weshy_media::default_camera_device());
                device_picker(ui, "Device", icons::PHOTO_CAMERA, &options, &mut form.camera.device, &mut form.camera_custom, &hint);
                if scanned && options.is_empty() {
                    caption(ui, "No cameras found — type the device instead");
                }
                if form.camera_custom || weshy_media::is_network_source(&form.camera.device) {
                    caption(ui, PHONE_HELP);
                }
                caption(ui, "Resolution");
                let camera = &mut form.camera;
                let mut resolution = (camera.width, camera.height);
                if ui.add(SegmentedButton::new(&mut resolution, &RESOLUTIONS)).changed() {
                    (camera.width, camera.height) = resolution;
                }
                caption(ui, "Frame rate");
                ui.add(SegmentedButton::new(&mut camera.fps, &FRAME_RATES));
                if live {
                    caption(ui, "Stop the camera to change these");
                }
            });
        },
    );
    if refresh {
        frame.actions.push(Action::RefreshDevices);
    }
}

// ── Output ─────────────────────────────────────────────────────────────────────

pub(super) fn output(ui: &mut Ui, frame: &mut Frame<'_>) {
    let live = frame.is_live();
    let streaming = live && frame.snapshot.output_active;
    let loopback = frame.devices.map_or(LoopbackStatus::Unsupported, |d| d.loopback);
    let mut options = frame.devices.map(|d| device_options(&d.outputs, true)).unwrap_or_default();
    if loopback == LoopbackStatus::NotLoaded {
        // The device the modprobe command below creates.
        options.push((DEFAULT_LOOPBACK.to_owned(), format!("{DEFAULT_LOOPBACK} · needs v4l2loopback")));
    }
    options.push((DEFAULT_STREAM.to_owned(), STREAM_LABEL.to_owned()));
    let scanning = frame.scanning;
    let form = &mut *frame.form;
    let (mut copied, mut refresh) = (false, false);
    section_with(
        ui,
        icons::CAST,
        "Output",
        |ui| refresh = refresh_button(ui, scanning),
        |ui| {
            ui.add_enabled_ui(!live, |ui| {
                let kind = match OutputTarget::parse(form.output_target.trim()) {
                    OutputTarget::V4l2(_) => "Appear as a webcam in other apps",
                    OutputTarget::Stream(_) => "Stream to OBS as a media source",
                };
                switch_row(ui, icons::SENSORS, "Virtual camera", kind, &mut form.output_enabled);
                ui.add_enabled_ui(form.output_enabled, |ui| {
                    device_picker(
                        ui,
                        "Target",
                        icons::CAST,
                        &options,
                        &mut form.output_target,
                        &mut form.output_custom,
                        "/dev/video10 or udp://…",
                    );
                });
            });
            let wants_loopback = matches!(OutputTarget::parse(form.output_target.trim()), OutputTarget::V4l2(_));
            if loopback == LoopbackStatus::NotLoaded && wants_loopback {
                copied = loopback_help(ui);
            }
            if streaming {
                ui.add(Chip::status("Sending frames", true).icon(icons::GRAPHIC_EQ));
            }
        },
    );
    if copied {
        frame.actions.push(Action::Notify("Command copied — run it in a terminal, then refresh".into()));
    }
    if refresh {
        frame.actions.push(Action::RefreshDevices);
    }
}

/// Dropdown label of the default network stream.
const STREAM_LABEL: &str = "OBS media source (UDP stream)";

/// Explains how to load v4l2loopback, with the command and a copy button. Returns whether it was copied.
fn loopback_help(ui: &mut Ui) -> bool {
    let colors = theme::colors(ui.ctx());
    let mut copied = false;
    egui::Frame::new().fill(colors.tertiary_container).corner_radius(shape::MEDIUM).inner_margin(Margin::same(spacing::MD as i8)).show(
        ui,
        |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = spacing::SM;
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(size::ICON), Sense::hover());
                paint::icon(ui.painter(), rect.center(), icons::WARNING, size::ICON, colors.on_tertiary_container, false);
                ui.label(TypeRole::TitleSmall.rich("v4l2loopback isn't loaded", colors.on_tertiary_container));
            });
            ui.add_space(spacing::XS);
            let text = "Other apps can only see weshy as a webcam once the virtual camera driver is loaded. Run:";
            ui.add(Label::new(TypeRole::BodyMedium.rich(text, colors.on_tertiary_container)).wrap());
            ui.add_space(spacing::XS);
            egui::Frame::new()
                .fill(colors.surface_container_lowest)
                .corner_radius(shape::SMALL)
                .inner_margin(Margin::same(spacing::SM as i8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                        let text_width = ui.available_width() - size::TARGET;
                        ui.allocate_ui(vec2(text_width, 0.0), |ui| {
                            ui.add(Label::new(TypeRole::BodySmall.rich(LOOPBACK_MODPROBE, colors.on_surface).monospace()).wrap());
                        });
                        if ui.add(IconButton::new(icons::CONTENT_COPY, "Copy command")).clicked() {
                            ui.ctx().copy_text(LOOPBACK_MODPROBE.to_owned());
                            copied = true;
                        }
                    });
                });
            ui.add_space(spacing::XS);
            ui.label(TypeRole::BodySmall.rich("Or pick the OBS media source target instead.", colors.on_tertiary_container));
        },
    );
    copied
}
