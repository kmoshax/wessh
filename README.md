# wessh

[![CI](https://github.com/kmoshax/wessh/actions/workflows/ci.yml/badge.svg)](https://github.com/kmoshax/wessh/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kmoshax/wessh)](https://github.com/kmoshax/wessh/releases/latest)

*wessh* (وشي) is Arabic for "my face". Real-time face swap in Rust. A lighter, faster take on Deep-Live-Cam: one native binary,
no Python, no PyTorch and no multi-gigabyte environment to build.

## Quick start

Download the file for your platform from [Releases](https://github.com/kmoshax/wessh/releases):

- **Linux:** `wessh-linux-x86_64.AppImage` (or `aarch64` for ARM). Make it executable in your file manager and open it. No distribution-specific package or administrator access is needed.
- **Windows:** `wessh-windows-x64-setup.exe`. Install, then open wessh from the Start menu.
- **macOS:** open the DMG and drag wessh into Applications. The bundle currently uses an ad-hoc signature, not Apple notarization.

The first-run screen detects your hardware and downloads the runtime and models once, with pause and resume. Video tools download inside the app when needed.

Linux AppImages include the X11/Wayland window libraries, curl and HTTPS certificates. They target glibc 2.35 or newer and use the host's graphics drivers. The release workflow tests graphical startup on Debian, Fedora and Arch. Alpine/musl is not supported. If FUSE is unavailable, run `./wessh-linux-x86_64.AppImage --appimage-extract-and-run`. A virtual webcam on Linux still requires the host's v4l2loopback kernel module; image processing and the app preview do not.

Portable archives and the command-line installers in `scripts/` remain available for terminal users. `wessh doctor` diagnoses runtime and model installation.

## Features

- Live webcam swap with an egui preview, per-stage timings, and virtual camera output
  (`/dev/videoN` via v4l2loopback on Linux, or `udp://` for OBS on any OS)
- Image and video modes; video keeps the audio and encodes with NVENC / VideoToolbox when available
- Swap models: inswapper 128 (fp16, the default), HyperSwap 1a / 1b / 1c 256
- GFPGAN 1.4 face restoration, applied only as far as the original face's detail goes
- One Euro landmark stabilisation, so the live swap doesn't jitter
- Execution providers picked automatically: TensorRT (fp16, engine + timing cache) → CUDA → CPU
- Self-contained installation: everything lives in one data directory, nothing is installed system-wide

## Why it's fast

- Pre/post-processing only touches the aligned crop and the face's bounding box, never whole-frame
  copies. Kernels are parallelised with rayon.
- Capture, inference and output run on separate threads. The worker always takes the *newest*
  frame and drops stale ones, so latency never builds up.
- Video decode, swap and encode are pipelined through bounded channels.
- ONNX Runtime is loaded dynamically: the binary stays small, and CUDA / TensorRT are a drop-in.
- Source embeddings, feather masks and inswapper's embedding map are computed once and cached.

## wessh vs Deep-Live-Cam

|                           | wessh                                        | Deep-Live-Cam                                                     |
| ------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| Install steps             | download, `wessh setup`, run                 | Python 3.10 + git + ffmpeg + (Windows) VS build tools, clone, venv, `pip install`, download models by hand, CUDA toolkit + cuDNN for GPU |
| Application download      | ≈ 3.5 MB archive (7 MB binary, Linux x86-64) | TODO (clone + `pip install`)                                      |
| Runtime + models, CPU     | ≈ 460 MB (ONNX Runtime 24 MB + models)      | TODO GB                                                           |
| Runtime + models, NVIDIA  | ≈ 1.9 GB download, ≈ 2.8 GB on disk         | TODO GB                                                           |
| Live latency, CPU         | TODO ms / frame                             | TODO ms / frame                                                   |
| Live latency, NVIDIA GPU  | TODO ms / frame                             | TODO ms / frame                                                   |
| System-wide changes       | none (one data directory)                   | Python packages, CUDA toolkit                                     |

Latency is measured with `wessh bench` (per-stage timings) on the same machine and source/target
images; see [CONTRIBUTING.md](CONTRIBUTING.md#benchmarks).

## Commands

Run `wessh <command> --help` for every option.

```sh
wessh                                              # desktop app
wessh image -s me.jpg -t photo.jpg -o out.jpg      # swap faces in an image
wessh video -s me.jpg -t clip.mp4 -o out.mp4
wessh cam   -s me.jpg --out /dev/video10 --mirror  # headless live -> virtual camera
wessh cam   -s me.jpg --out udp://127.0.0.1:27000  # headless live -> OBS media source
wessh bench -s me.jpg -t photo.jpg -n 100          # per-stage latency
wessh models                                       # download the curated models
wessh setup                                        # install runtime, CUDA / TensorRT libraries, models
wessh doctor                                       # diagnose the installation
```

Global options:

| Option                                                  | Meaning                                          |
| ------------------------------------------------------- | ------------------------------------------------ |
| `--device auto\|tensorrt\|cuda\|cpu`                     | inference backend (default `auto`)               |
| `--enhance true\|false`                                 | GFPGAN face restoration (default on)             |
| `--many-faces`                                          | swap every face, not only the largest            |
| `--detector-size 320\|480\|640`                          | detector input; smaller is faster                |
| `--ort <path>`                                          | use a specific ONNX Runtime library              |

`wessh setup` options:

| Option                           | Meaning                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `--dry-run`                      | print the plan with download sizes, change nothing                   |
| `--flavour cpu\|cuda12\|cuda13\|coreml` | override the detected runtime build                            |
| `--no-tensorrt`                  | skip TensorRT on NVIDIA (saves a 3.7 GB download; fastest backend)   |
| `--force`                        | reinstall components that are already present                       |
| `--no-models`                    | skip the models (the app downloads them on first start)              |
| `--ort-version <x.y.z>`          | ONNX Runtime release to install (default: the tested one)            |

Setup is idempotent (finished components are skipped), downloads resume after interruption,
and every file is checked against the SHA-256 its source publishes.

### How the runtime flavour is chosen

| Machine                                  | Flavour  | Installed                                         |
| ---------------------------------------- | -------- | ------------------------------------------------- |
| NVIDIA, driver 580 or newer (x86-64)     | `cuda13` | ONNX Runtime GPU + CUDA 13, cuBLAS, cuFFT, cuRAND, cuDNN 9 |
| NVIDIA, driver 525–579 (x86-64)          | `cuda12` | ONNX Runtime GPU + CUDA 12 libraries              |
| Apple Silicon                            | `coreml` | ONNX Runtime for macOS arm64                      |
| anything else                            | `cpu`    | ONNX Runtime CPU                                  |

The CUDA libraries come from NVIDIA's official redistributable wheels on PyPI, so no CUDA
toolkit is needed — only the NVIDIA driver.

## Files and environment

Everything wessh downloads lives in one directory: `$WESSH_HOME`, or by default
`~/.local/share/wessh` (Linux), `~/Library/Application Support/wessh` (macOS),
`%LOCALAPPDATA%\wessh` (Windows).

```text
<data dir>/ort/lib/          ONNX Runtime and its provider plugins
<data dir>/ort/lib/cuda/     CUDA, cuBLAS, cuFFT, cuRAND, cuDNN (and TensorRT) libraries
<data dir>/models/           ONNX models
<data dir>/trt_cache/        TensorRT engine and timing caches
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

### Virtual camera on Linux

```sh
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="wessh" exclusive_caps=1
wessh cam -s me.jpg --out /dev/video10
```

`wessh doctor` tells you whether the module is installed and loaded.

## Platform notes

- **Intel Macs**: Microsoft no longer publishes ONNX Runtime for macOS x86-64. Install it with
  `brew install onnxruntime` and pass `--ort "$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"`.
- **Windows without NVIDIA**: runs on the CPU. DirectML builds of ONNX Runtime stop at 1.24,
  older than the 1.28 wessh needs.
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
