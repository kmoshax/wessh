![wessh — real-time face swap, built in Rust](docs/assets/banner.svg)

[![CI](https://github.com/kmoshax/wessh/actions/workflows/ci.yml/badge.svg)](https://github.com/kmoshax/wessh/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kmoshax/wessh)](https://github.com/kmoshax/wessh/releases/latest)

**wessh** (وشي, “my face”) is an educational, native Rust app for real-time face swapping. No Python environment needed.

## Quick start

Download from [Releases](https://github.com/kmoshax/wessh/releases), then open:

- **Linux:** AppImage — make it executable and launch it.
- **Windows:** setup EXE — install and open from the Start menu.
- **macOS:** DMG — drag wessh into Applications.

First launch sets up your runtime and models. Downloads happen once and can resume.

## Features

- Live webcam preview, plus image and video face swapping.
- Face restoration and stable tracking.
- Automatic GPU acceleration with a CPU fallback.
- Local processing, with runtime and models kept in one data directory.

<details>
<summary>Performance notes</summary>

- Face crops are processed in parallel without copying whole frames.
- Capture, inference and output run on separate threads; stale frames are dropped.
- Video decoding and encoding are pipelined, with NVENC / VideoToolbox when available.
- ONNX Runtime loads dynamically; embeddings, masks and TensorRT engines are cached.

Measure your hardware with `wessh bench`; see [benchmark instructions](CONTRIBUTING.md#benchmarks).

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

<details>
<summary>Installation details</summary>

- **Linux:** AppImages bundle X11/Wayland libraries, curl and HTTPS certificates. They require glibc 2.35+ and your graphics driver; graphical startup is tested on Debian, Fedora and Arch. Alpine/musl is unsupported. Without FUSE, run `./wessh-linux-x86_64.AppImage --appimage-extract-and-run`.
- **macOS:** the app is ad-hoc signed, not Apple-notarized.
- Portable archives and CLI installers are available in `scripts/`. Run `wessh doctor` to check your installation.
- Linux virtual webcam output requires the host's v4l2loopback kernel module. App preview and image processing do not.

</details>

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

## Educational use & disclaimer

wessh is an educational and experimental project, provided **as is**, without warranties. Use only media you own or have permission to use, with the consent of the people depicted. Do not use it for impersonation, deception, harassment or non-consensual content. You are responsible for your use and outputs; third-party models have their own licenses and usage terms.

## License

Copyright (C) 2026 kmoshax.

wessh is free software, licensed under the [GNU Affero General Public License v3.0 only](LICENSE).
You may use, study, modify and share it, but anything you distribute or offer to others over a
network that is built on wessh must be released, in full source, under the same license.
