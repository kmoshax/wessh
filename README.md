# wajh

Real-time face swap written in Rust. A lighter, faster take on Deep-Live-Cam:
a single native binary with no Python, no PyTorch and no multi-GB environment.

## Features

- Live webcam swap with an egui preview, stats per stage, and virtual camera output
  (`/dev/videoN` via v4l2loopback, or `udp://` for OBS on any OS)
- Image and video modes (video keeps the audio, and uses NVENC / VideoToolbox when available)
- Swap models: inswapper 128 (fp16), HyperSwap 1a / 1b / 1c 256
- GPEN-256 face enhancer (much faster than GFPGAN)
- One Euro landmark stabilisation, so the live swap doesn't jitter
- Execution providers picked automatically: TensorRT (fp16, engine + timing cache) → CUDA → CPU
- Models are downloaded on first use into `~/.local/share/wajh/models`

## Why it's fast

- Pre/post-processing only touches the aligned crop and the face's bounding box, never whole-frame copies.
  Kernels are parallelised with rayon.
- Capture, inference and output run on separate threads. The worker always takes the *newest* frame and
  drops stale ones, so latency never builds up.
- Video decode, swap and encode are pipelined through bounded channels.
- ONNX Runtime is loaded dynamically: the binary stays small, and CUDA/TensorRT are an optional drop-in.
- Source embeddings, feather masks and inswapper's embedding map are computed once and cached.

## Setup

Requires `ffmpeg` on `PATH` and an ONNX Runtime shared library (1.26+).
wajh looks for the library in these places:

1. `--ort <path>` or `ORT_DYLIB_PATH`
2. next to the executable (or in `./ort/`)
3. `~/.local/share/wajh/ort/lib/`

For NVIDIA GPUs, use an ONNX Runtime GPU build. Its `libonnxruntime_providers_cuda.so` and
`libonnxruntime_providers_tensorrt.so` must sit next to `libonnxruntime.so`, and CUDA 13 + cuDNN 9
(+ TensorRT) must be on the library path.

Virtual camera on Linux:

```sh
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="wajh" exclusive_caps=1
```

## Usage

```sh
wajh                                              # desktop app
wajh image -s me.jpg -t photo.jpg -o out.jpg
wajh video -s me.jpg -t clip.mp4 -o out.mp4 --enhance
wajh cam   -s me.jpg --out /dev/video10 --mirror  # headless live -> virtual camera
wajh bench -s me.jpg -t photo.jpg                 # per-stage latency
```

Global options: `--model inswapper|hyperswap1a|hyperswap1b|hyperswap1c`, `--device auto|tensorrt|cuda|cpu`,
`--enhance`, `--many-faces`, `--detector-size 320|480|640`.

## Roadmap

- GPU pre/post-processing (CUDA kernels + IoBinding) so frames never leave the GPU
- Optical-flow tracking between detections
- XSeg occlusion and BiSeNet region masks, mouth mask
- Native virtual camera on Windows 11 (`MFCreateVirtualCamera`) and macOS (Camera Extension)
- Portable release builds via cargo-dist
