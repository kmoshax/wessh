![wessh — real-time face swap, built in Rust](docs/assets/banner.svg)

[![CI](https://github.com/kmoshax/wessh/actions/workflows/ci.yml/badge.svg)](https://github.com/kmoshax/wessh/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kmoshax/wessh)](https://github.com/kmoshax/wessh/releases/latest)

**wessh** (وش, “face”) is an educational, native Rust app for real-time face swapping. No Python environment needed.

> [!CAUTION]
> **Educational and experimental project.** Provided as is, without warranties. Use media with permission and the consent of the people depicted. Do not use it for deception, harassment or non-consensual content. You are responsible for your use and outputs; third-party models have their own terms.

## Demos

<table>
  <tr>
    <td width="70%" align="center">
      <img src="docs/assets/demo.gif" width="480" alt="MrBeast face-swap demo — twelve edited clips">
    </td>
    <td width="30%" align="center">
      <img src="docs/assets/mrbeast.png" width="203" alt="MrBeast source portrait used for the demo">
      <br><sub>Source face - MrBeast</sub>
    </td>
  </tr>
  <tr>
    <td width="70%" align="center">
      <img src="docs/assets/demo-hitler.gif" width="480" alt="Adolf Hitler face-swap demo — twelve edited clips">
    </td>
    <td width="30%" align="center">
      <img src="docs/assets/hitler.jpg" width="203" alt="Historical portrait used as the source face for the additional demo">
      <br><sub>Source face - Adolf Hitler</sub>
    </td>
  </tr>
</table>

_12 clips - Edited footage - [Sources & credits](docs/demo-sources.md)_

## Quick start

Download from [Releases](https://github.com/kmoshax/wessh/releases), then open:

- **Linux:** AppImage — make it executable and launch it.
- **Windows:** setup EXE — install and open from the Start menu.
- **macOS:** DMG — drag wessh into Applications.

First launch installs your runtime and models for enabled features. Face restoration and masks download when you enable them; ffmpeg is fetched the first time a video file or phone camera needs it. Downloads happen once and can resume.

## Updates

Use the update button in the app bar to check stable releases, download an update, then install and restart. The updater preserves your installation format, models and runtime cache; it waits for processing jobs to finish before restarting.

## Features

- Live webcam preview, plus image and video face swapping.
- Face restoration and stable tracking.
- Automatic GPU acceleration on NVIDIA, AMD, Intel and Apple GPUs, with a CPU fallback.
- Local processing, with runtime and models kept in one data directory.

<details>
<summary>Performance notes</summary>

- Face crops are processed in parallel without copying whole frames.
- Capture, inference and output run on separate threads; stale frames are dropped.
- Live mode runs the swap network beside the frame loop: every frame is pasted at camera rate
  with the newest swap, re-aligned to where the face is now, and a face that hasn't changed
  isn't swapped again.
- Landmarks are followed by optical flow, steady on a still face without lagging a moving one.
- Video decoding and encoding are pipelined, with NVENC / VideoToolbox when available.
- ONNX Runtime loads dynamically; embeddings, masks and compiled GPU kernels are cached.

Measured with `wessh bench` (a 2079×2739 photo, inswapper, restoration off) on a laptop
RTX 4050 6 GB, driver 615, Linux:

| Backend                      | Frame time | 95th percentile |
| ---------------------------- | ---------- | --------------- |
| CUDA 13 + cuDNN              | 33.0 ms    | 43.6 ms         |
| TensorRT for RTX             | 21.5 ms    | 23.8 ms         |
| TensorRT for RTX, live mode  | 11.3 ms    | 14.5 ms         |

Live mode is the async swap above, on a still face; with the face restored by GFPGAN the swap
alone takes 126 ms (CUDA: 193 ms). Measure your hardware with `wessh bench`; see
[benchmark instructions](CONTRIBUTING.md#benchmarks).

</details>

## Commands

Run `wessh <command> --help` for every option.

```sh
wessh                                              # desktop app
wessh image -s me.jpg -t photo.jpg -o out.jpg      # swap faces in an image
wessh video -s me.jpg -t clip.mp4 -o out.mp4
wessh cam   -s me.jpg --out /dev/video10 --mirror  # headless live -> virtual camera
wessh cam   -s me.jpg --out udp://127.0.0.1:27000  # headless live -> OBS media source
wessh bench -s me.jpg -t photo.jpg -n 100          # per-stage latency
wessh models                                       # download models for enabled features
wessh setup                                        # install runtime, CUDA libraries and enabled models
wessh doctor                                       # diagnose the installation
```

Global options:

| Option                                                  | Meaning                                          |
| ------------------------------------------------------- | ------------------------------------------------ |
| `--device auto\|rtx\|tensorrt\|cuda\|webgpu\|cpu`        | inference backend (default `auto`)               |
| `--enhance true\|false`                                 | GFPGAN face restoration (default off)            |
| `--many-faces`                                          | swap every face, not only the largest            |
| `--detector-size 320\|480\|640`                          | detector input; smaller is faster                |
| `--ort <path>`                                          | use a specific ONNX Runtime library              |

`wessh setup` options:

| Option                           | Meaning                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `--dry-run`                      | print the plan with download sizes, change nothing                   |
| `--flavour cpu\|cuda12\|cuda13\|coreml\|rtx\|webgpu` | override the detected runtime build (default: `rtx` on RTX 30-series and newer with driver 580+, CUDA on older NVIDIA GPUs, `webgpu` on AMD / Intel) |
| `--force`                        | reinstall components that are already present                       |
| `--no-models`                    | skip the models (the app downloads them on first start)              |
| `--ort-version <x.y.z>`          | ONNX Runtime release to install (default: the tested one)            |

Setup is idempotent (finished components are skipped) and downloads resume after interruption.
From NVIDIA's wheels only the libraries wessh loads are downloaded (HTTP range requests), each
checked against the SHA-256 the wheel lists for it; every other file is checked against the
SHA-256 its source publishes.

### How the runtime flavour is chosen

| Machine                                          | Flavour  | Installed                                              | Download |
| ------------------------------------------------ | -------- | ------------------------------------------------------ | -------- |
| NVIDIA RTX 30-series or newer, driver 525+       | `rtx`    | ONNX Runtime + NVIDIA TensorRT for RTX                 | 104 MB   |
| other NVIDIA GPUs, driver 580+ (x86-64)          | `cuda13` | ONNX Runtime GPU + CUDA 13, cuBLAS, cuRAND, cuDNN 9    | 1.3 GB   |
| other NVIDIA GPUs, driver 525–579 (x86-64)       | `cuda12` | ONNX Runtime GPU + CUDA 12 libraries                   | 1.8 GB   |
| AMD or Intel GPU (Windows; Linux with Vulkan)    | `webgpu` | ONNX Runtime + WebGPU                                  | 16 MB    |
| Apple Silicon                                    | `coreml` | ONNX Runtime for macOS arm64                           | 9 MB     |
| anything else                                    | `cpu`    | ONNX Runtime CPU                                       | 9 MB     |

The face models add 469 MB (GFPGAN restoration another 366 MB, when you turn it on). The NVIDIA
libraries come from NVIDIA's official redistributable wheels on PyPI, so no CUDA toolkit is
needed — only the driver. TensorRT for RTX compiles its kernels for your GPU on the first start
(about 10 s, shown at the end of setup) and reuses them afterwards.

## Files and environment

Everything wessh downloads lives in one directory: `$WESSH_HOME`, or by default
`~/.local/share/wessh` (Linux), `~/Library/Application Support/wessh` (macOS),
`%LOCALAPPDATA%\wessh` (Windows).

```text
<data dir>/ort/lib/          ONNX Runtime and its provider plugins
<data dir>/ort/lib/cuda/     CUDA, cuBLAS, cuRAND, cuDNN libraries (CUDA builds)
<data dir>/ort/lib/plugins/  TensorRT for RTX or WebGPU plugin
<data dir>/rtx_cache/        TensorRT for RTX kernel cache
<data dir>/models/           ONNX models
<data dir>/trt_cache/        TensorRT engine caches (only with an `--ort` runtime that has TensorRT)
```

Uninstalling is deleting that directory and the binary.

wessh looks for ONNX Runtime in this order: `--ort <path>`, `ORT_DYLIB_PATH`, next to the
executable (also `./ort/lib` and `./lib`), then `<data dir>/ort/lib`.

| Variable                         | Effect                                                   |
| -------------------------------- | -------------------------------------------------------- |
| `WESSH_HOME`       | data directory                                           |
| `ORT_DYLIB_PATH`                 | ONNX Runtime library to load                             |
| `WESSH_CUDA_DIR`                  | directory of CUDA libraries to preload                   |
| `WESSH_THREADS`                   | CPU inference threads (default: one per physical core)   |
| `WESSH_FFMPEG`                   | ffmpeg executable to run                                 |
| `WESSH_TIMEWARP=0`               | live mode: swap every frame in line (for measuring)      |

### Virtual camera on Linux

```sh
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="wessh" exclusive_caps=1
wessh cam -s me.jpg --out /dev/video10
```

`wessh doctor` tells you whether the module is installed and loaded.

## Platform notes

<details>
<summary>Installation details</summary>

- **Linux:** AppImages bundle X11/Wayland libraries, curl and HTTPS certificates. They require glibc 2.35+ and your graphics driver; graphical startup is tested on Debian, Fedora and Arch. Alpine/musl is unsupported. Without FUSE, run `./wessh-linux-x86_64.AppImage --appimage-extract-and-run`.
- **macOS:** the app is ad-hoc signed, not Apple-notarized.
- Portable archives and CLI installers are available in `scripts/`. Run `wessh doctor` to check your installation.
- Linux virtual webcam output requires the host's v4l2loopback kernel module. App preview and image processing do not.

</details>

- **Intel Macs**: Microsoft no longer publishes ONNX Runtime for macOS x86-64. Install it with
  `brew install onnxruntime` and pass `--ort "$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"`.
- **AMD / Intel GPUs on Linux** need the Vulkan loader (`libvulkan.so.1`, packaged as
  `vulkan-icd-loader` / `libvulkan1`) for WebGPU; without it wessh runs on the CPU.
- **Linux arm64 with NVIDIA** (Jetson and similar): ONNX Runtime publishes no GPU build, so
  `wessh setup` installs the CPU one; point `--ort` at a JetPack build to use the GPU.

## Building from source

```sh
cargo build --release
./target/release/wessh setup
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Roadmap

- GPU pre/post-processing (CUDA kernels + IoBinding) so frames never leave the GPU
- Optical-flow tracking between detections
- XSeg occlusion and BiSeNet region masks, mouth mask
- CoreML execution provider on Apple Silicon
- Native virtual camera on Windows 11 (`MFCreateVirtualCamera`) and macOS (Camera Extension)

## License

Copyright (C) 2026 kmoshax.

wessh is free software, licensed under the [GNU Affero General Public License v3.0 only](LICENSE).
You may use, study, modify and share it, but anything you distribute or offer to others over a
network that is built on wessh must be released, in full source, under the same license.
