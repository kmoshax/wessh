# wajh

Real-time face swap in Rust. A lighter, faster take on Deep-Live-Cam: one native binary,
no Python, no PyTorch and no multi-gigabyte environment to build.

## Quick start

1. **Download** the archive for your platform from
   [Releases](https://github.com/kmosha/wajh/releases) and unpack it, or use the installer:

   ```sh
   # Linux / macOS
   curl -fsSL https://raw.githubusercontent.com/kmosha/wajh/main/scripts/install.sh | sh
   ```

   ```powershell
   # Windows
   irm https://raw.githubusercontent.com/kmosha/wajh/main/scripts/install.ps1 | iex
   ```

2. **Set up** the machine. `wajh setup` detects your GPU and driver, installs the matching
   ONNX Runtime build, the CUDA / cuDNN libraries it needs (NVIDIA only, no CUDA toolkit
   install required) and the default models:

   ```sh
   wajh setup --dry-run   # optional: see what would be downloaded, and how much
   wajh setup
   ```

3. **Run** it:

   ```sh
   wajh
   ```

If anything misbehaves, `wajh doctor` reports what is installed and prints the exact command
that fixes each problem. The only thing `wajh setup` can't install for you is ffmpeg (used for
the camera and video modes); it prints the command for your OS if ffmpeg is missing.

## Features

- Live webcam swap with an egui preview, per-stage timings, and virtual camera output
  (`/dev/videoN` via v4l2loopback on Linux, or `udp://` for OBS on any OS)
- Image and video modes; video keeps the audio and encodes with NVENC / VideoToolbox when available
- Swap models: inswapper 128 (fp16), HyperSwap 1a / 1b / 1c 256
- GPEN-256 face enhancer (much faster than GFPGAN)
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

## wajh vs Deep-Live-Cam

|                           | wajh                                        | Deep-Live-Cam                                                     |
| ------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| Install steps             | download, `wajh setup`, run                 | Python 3.10 + git + ffmpeg + (Windows) VS build tools, clone, venv, `pip install`, download models by hand, CUDA toolkit + cuDNN for GPU |
| Application download      | ≈ 3.5 MB archive (7 MB binary, Linux x86-64) | TODO (clone + `pip install`)                                      |
| Runtime + models, CPU     | ≈ 460 MB (ONNX Runtime 24 MB + models)      | TODO GB                                                           |
| Runtime + models, NVIDIA  | ≈ 1.9 GB download, ≈ 2.8 GB on disk         | TODO GB                                                           |
| Live latency, CPU         | TODO ms / frame                             | TODO ms / frame                                                   |
| Live latency, NVIDIA GPU  | TODO ms / frame                             | TODO ms / frame                                                   |
| System-wide changes       | none (one data directory)                   | Python packages, CUDA toolkit                                     |

Latency is measured with `wajh bench` (per-stage timings) on the same machine and source/target
images; see [CONTRIBUTING.md](CONTRIBUTING.md#benchmarks).

## Commands

Run `wajh <command> --help` for every option.

```sh
wajh                                              # desktop app
wajh image -s me.jpg -t photo.jpg -o out.jpg      # swap faces in an image
wajh video -s me.jpg -t clip.mp4 -o out.mp4
wajh cam   -s me.jpg --out /dev/video10 --mirror  # headless live -> virtual camera
wajh cam   -s me.jpg --out udp://127.0.0.1:27000  # headless live -> OBS media source
wajh bench -s me.jpg -t photo.jpg -n 100          # per-stage latency
wajh models                                       # download the models the options need
wajh setup                                        # install runtime, CUDA libraries, models
wajh doctor                                       # diagnose the installation
```

Global options:

| Option                                                  | Meaning                                          |
| ------------------------------------------------------- | ------------------------------------------------ |
| `--model inswapper\|hyperswap1a\|hyperswap1b\|hyperswap1c` | swap model (default `hyperswap1a`)            |
| `--device auto\|tensorrt\|cuda\|cpu`                     | inference backend (default `auto`)               |
| `--enhance true\|false`                                 | GPEN face restoration (default on)               |
| `--many-faces`                                          | swap every face, not only the largest            |
| `--detector-size 320\|480\|640`                          | detector input; smaller is faster                |
| `--ort <path>`                                          | use a specific ONNX Runtime library              |

`wajh setup` options:

| Option                           | Meaning                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `--dry-run`                      | print the plan with download sizes, change nothing                   |
| `--flavour cpu\|cuda12\|cuda13\|coreml` | override the detected runtime build                            |
| `--tensorrt`                     | also install TensorRT (NVIDIA; large download, fastest backend)      |
| `--force`                        | reinstall components that are already present                       |
| `--no-models`                    | skip the models (they otherwise download on first use)               |
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

Everything wajh downloads lives in one directory: `$WAJH_HOME`, or by default
`~/.local/share/wajh` (Linux), `~/Library/Application Support/wajh` (macOS),
`%LOCALAPPDATA%\wajh` (Windows).

```text
<data dir>/ort/lib/          ONNX Runtime and its provider plugins
<data dir>/ort/lib/cuda/     CUDA, cuBLAS, cuFFT, cuRAND, cuDNN (and TensorRT) libraries
<data dir>/models/           ONNX models
<data dir>/trt_cache/        TensorRT engine and timing caches
```

Uninstalling is deleting that directory and the binary.

wajh looks for ONNX Runtime in this order: `--ort <path>`, `ORT_DYLIB_PATH`, next to the
executable (also `./ort/lib` and `./lib`), then `<data dir>/ort/lib`.

| Variable                         | Effect                                                   |
| -------------------------------- | -------------------------------------------------------- |
| `WAJH_HOME`                      | data directory                                           |
| `ORT_DYLIB_PATH`                 | ONNX Runtime library to load                             |
| `WAJH_CUDA_DIR`                  | directory of CUDA libraries to preload                   |
| `WAJH_THREADS`                   | CPU inference threads (default: one per physical core)   |
| `WAJH_FFMPEG`, `WAJH_FFPROBE`    | ffmpeg / ffprobe executables                             |

### Virtual camera on Linux

```sh
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="wajh" exclusive_caps=1
wajh cam -s me.jpg --out /dev/video10
```

`wajh doctor` tells you whether the module is installed and loaded.

## Platform notes

- **Intel Macs**: Microsoft no longer publishes ONNX Runtime for macOS x86-64. Install it with
  `brew install onnxruntime` and pass `--ort "$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"`.
- **Windows without NVIDIA**: runs on the CPU. DirectML builds of ONNX Runtime stop at 1.24,
  older than the 1.28 wajh needs.
- **Linux arm64 with NVIDIA** (Jetson and similar): ONNX Runtime publishes no GPU build, so
  `wajh setup` installs the CPU one; point `--ort` at a JetPack build to use the GPU.

## Building from source

```sh
cargo build --release
./target/release/wajh setup
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Roadmap

- GPU pre/post-processing (CUDA kernels + IoBinding) so frames never leave the GPU
- Optical-flow tracking between detections
- XSeg occlusion and BiSeNet region masks, mouth mask
- CoreML execution provider on Apple Silicon
- Native virtual camera on Windows 11 (`MFCreateVirtualCamera`) and macOS (Camera Extension)
