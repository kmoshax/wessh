# Rust ML Inference, GPU Ecosystem and Performance Practices for a Real-Time ONNX Face-Swap Pipeline (state as of Sept 2026)

Pipeline under consideration: detection (SCRFD / RetinaFace det_10g) → embedding (ArcFace w600k_r50) → swap (inswapper_128) → paste-back blend → enhancer (GFPGAN / CodeFormer / GPEN). Targets: NVIDIA, AMD, Intel, Apple GPUs and CPU.

## 1. `ort` crate (pykeio/ort): version, ONNX Runtime version, EPs, linking, IoBinding, FP16, pitfalls

### Takeaway
`ort` is still the only production-grade way to run arbitrary ONNX models (inswapper, SCRFD, GFPGAN) from Rust across all GPU vendors, but 2.0 has **never gone stable**: the latest is **v2.0.0-rc.13 (28 Jul 2026)**, wrapping **ONNX Runtime 1.28**, shipping prebuilt binaries for **CUDA 13 only**. Expect breaking API changes between RCs; pin exact versions.

### Cited Findings
- Latest release is v2.0.0-rc.13; it "skips ahead 4 ONNX Runtime versions to v1.28" — [newreleases.io: ort rc.13](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13)
- Release dates: rc.12 on 2026-03-05, rc.13 on 2026-07-28 — [newreleases.io / search summary](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13); docs.rs "latest" still showed rc.12 at time of indexing — [docs.rs ort](https://docs.rs/crate/ort/latest). (Note: a WebFetch summary of the GitHub releases page mislabeled years as 2023/2024; newreleases.io and docs.rs dates of 2026 are taken as correct.)
- 2.0.0 is not stable; still in release-candidate phase with ongoing breaking changes (module restructuring, EP enum renames, custom-op trait redesign) — [GitHub pykeio/ort releases](https://github.com/pykeio/ort/releases)
- rc.13: only CUDA 13 binaries shipped because ONNX Runtime deprecated CUDA 12 builds — [newreleases.io rc.13](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13)
- rc.13: EP structs (e.g. `ep::CUDA`) are now compile-time gated behind their Cargo features; `download-binaries` now errors **at link time** if no prebuilt binary contains all requested EPs (previously it silently fell back to CPU) unless `lax-feature-matching` is enabled — [newreleases.io rc.13](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13)
- rc.13 also: VSINPU EP added, DirectML EP API exposed, native `f16` support on nightly Rust, ort-candle backend updated (to candle 0.11 per notes), ort-tract updated to tract 0.23, Miri-verified soundness fixes, FreeBSD & Mac Catalyst support — [newreleases.io rc.13](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13)
- rc.12: "multiversioning" (runtime support for ONNX Runtime API v1.17–1.24), automatic device selection, CUDA 13 support; rc.11: ort-web, iOS/Android builds, Intel macOS prebuilt dropped; rc.10: alternative backends (tract, candle), Model Editor API, Compiler API, tensor copying — [GitHub pykeio/ort releases](https://github.com/pykeio/ort/releases)
- EPs referenced across releases/docs: CUDA, TensorRT, TensorRT-RTX, DirectML, CoreML, OpenVINO, ROCm, MIGraphX, QNN, WebGPU, NNAPI, Azure, VSINPU — [GitHub pykeio/ort releases](https://github.com/pykeio/ort/releases); [ort.pyke.io execution providers](https://ort.pyke.io/perf/execution-providers)
- Prebuilt (download-binaries) CUDA binaries require **CUDA ≥ 13.2 and cuDNN ≥ 9.23** (as stated on ort docs) on PATH; `copy-dylibs` (default on) copies EP dylibs next to the binary — [ort.pyke.io execution providers](https://ort.pyke.io/perf/execution-providers)
- Historically, prebuilt Microsoft binaries only cover CUDA/TensorRT on Win/Linux; other EPs (OpenVINO, MIGraphX, ROCm, QNN…) require building ONNX Runtime yourself — [lib.rs ort / docs summary](https://lib.rs/crates/ort)
- **Default silent fallback**: if an EP fails to register or lacks an op, execution silently falls back to the next EP / CPU; use `.error_on_failure()` to make failure fatal — [ort.pyke.io execution providers](https://ort.pyke.io/perf/execution-providers)
- WebGPU EP is marked experimental in ort and "may produce incorrect results/crashes" — [ort.pyke.io execution providers](https://ort.pyke.io/perf/execution-providers)
- Linking: static linking is "recommended" when the EP permits (`ORT_LIB_PATH`); `load-dynamic` loads `libonnxruntime` at runtime via `ort::init_from()` or `ORT_DYLIB_PATH`; compile-time dynamic linking via `ORT_PREFER_DYNAMIC_LINK=1` needs rpath/`LD_LIBRARY_PATH` care — [ort.pyke.io linking](https://ort.pyke.io/setup/linking)
- IoBinding: `session.create_binding()`, `bind_input`, `bind_output` (pre-allocated tensor) or `bind_output_to_device(name, &memory_info)`, then `session.run_binding(&binding)`; allocators can be created with `AllocationDevice::CUDA_PINNED` / CUDA device memory via `MemoryInfo` — [ort.pyke.io IoBinding](https://ort.pyke.io/perf/io-binding)
- IoBinding pitfalls: bound outputs are overwritten on the next run (copy out if needed); input data isn't guaranteed synchronized until `synchronize_inputs()` or `run_binding()`; re-binding only syncs that one input — [ort.pyke.io IoBinding](https://ort.pyke.io/perf/io-binding)
- IoBinding biggest gains: constant inputs kept on device (e.g. the source-face latent for inswapper), outputs that feed subsequent on-device ops, reused pre-allocated outputs — [ort.pyke.io IoBinding](https://ort.pyke.io/perf/io-binding)

### Inferences
- For a face-swap app, use `load-dynamic` to ship one binary that can pick different ONNX Runtime builds per platform (CUDA/TensorRT build on NVIDIA, a DirectML or WinML-supplied runtime on Windows AMD/Intel, CoreML build on macOS) — this is also the only practical route to non-prebuilt EPs like MIGraphX/OpenVINO.
- inswapper's source-embedding input (`source`, 1×512) is constant per session → bind once on device with IoBinding; bind the 1×3×128×128 `target` input and output to device buffers to feed paste-back on GPU.
- Always call `.error_on_failure()` in dev builds; silent CPU fallback is the #1 "why is it 3 FPS" cause in Python face-swap apps too.
- Pin `ort = "=2.0.0-rc.13"` because RC-to-RC changes are breaking.

### Gaps
- Could not verify whether ort rc.13 prebuilt binaries include TensorRT-RTX or WebGPU EPs by default for each OS (the per-platform EP widget on ort docs didn't render in fetch).
- Status of the ROCm EP (vs MIGraphX) in ONNX Runtime 1.28 not verified in this session; MIGraphX is the AMD EP Microsoft ships via Windows ML (below).
- No first-party measurement of ort (Rust) vs onnxruntime Python per-call overhead was found.

## 2. DirectML maintenance mode → Windows ML; Rust support

### Takeaway
DirectML is in maintenance mode (security fixes only). Microsoft's replacement is **Windows ML** (announced Build 2025, GA later in 2025), which ships an ONNX Runtime with CPU+DirectML in-box and **downloads vendor EPs on demand** (NVIDIA TensorRT-for-RTX, AMD MIGraphX & VitisAI, Intel OpenVINO, Qualcomm QNN, experimental WebGPU) on Windows 11 24H2+. There is no dedicated Rust crate for new Windows ML; the practical path from Rust is `ort` (DirectML EP, or `load-dynamic` against a WinML-provided runtime and registering plugin EPs) — unverified end-to-end.

### Cited Findings
- "DirectML is in maintenance mode… no new functionality or feature updates are planned"; security/compliance fixes continue — [microsoft/DirectML GitHub](https://github.com/microsoft/DirectML)
- New feature development moved to Windows ML for Windows-based ONNX Runtime deployments; Windows ML exposes the same ONNX Runtime APIs and selects EPs based on hardware — [ONNX Runtime DirectML EP docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html); [Windows Developer Blog, Introducing Windows ML (May 19 2025)](https://blogs.windows.com/windowsdeveloper/2025/05/19/introducing-windows-ml-the-future-of-machine-learning-development-on-windows/)
- Windows ML in-box EPs: CPU and DirectML (labelled "legacy"). Downloadable EPs via `ExecutionProviderCatalog` APIs on Windows 11 24H2 (build 26100)+: MIGraphX (AMD, RDNA3+), NvTensorRtRtx (NVIDIA RTX 30xx+, CUDA 12.5), OpenVINO (Intel), QNN (Qualcomm), VitisAI (AMD NPU), WebGPU (experimental, WinML 2.x only, `Microsoft.Windows.AI.MachineLearning` ≥ 2.4.66-preview) — [Microsoft Learn: Windows ML execution providers (updated 2026-09-22)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
- Distribution is via NuGet packages `Microsoft.WindowsAppSDK.ML` (1.8.x) / `Microsoft.Windows.AI.MachineLearning` (2.x) — [Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
- The `windows` crate exposes only legacy `Win32::AI::MachineLearning::WinML` bindings (old WinML API, not the new ORT-based Windows ML) — [windows-docs-rs WinML](https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/AI/MachineLearning/WinML/index.html)
- ort has a `directml` feature for DirectX 12 GPUs on Windows x86/x64 — [lib.rs / docs.rs ort](https://docs.rs/ort/latest/ort/)

### Inferences
- For Windows AMD/Intel GPUs today, DirectML via `ort` remains the simplest working choice (it still ships in-box and works); for NVIDIA on Windows, the new "TensorRT for RTX" EP (NvTensorRtRtx) is Microsoft's intended path and is also in ONNX Runtime upstream docs ([TensorRT RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)).
- Using Windows ML's EP catalog from Rust would require calling the WinRT/`ExecutionProviderCatalog` API (via `windows` crate projections if/when available) to get EP library paths, then registering them with ort's plugin/"auto device selection" APIs (rc.12) — plausible but not documented.

### Gaps
- No Rust crate or ort documentation explicitly supporting Windows ML / `ExecutionProviderCatalog` was found.

## 3. Alternatives: burn, candle, tract, wonnx, TensorRT bindings, cudarc

### Takeaway
None of the pure-Rust frameworks is a proven drop-in for inswapper/SCRFD/GFPGAN at ORT/TensorRT speed. Burn (0.20, Jan 2026) + the separate **burn-onnx 0.21** (Mar 2026) is the most credible pure-Rust path (code-gen from ONNX, CUDA/ROCm/Metal/Vulkan/WebGPU via CubeCL); candle-onnx is an interpreter, CPU-hardcoded in its main API and very slow on InsightFace models; tract is CPU-only; **wonnx is archived (May 2025)**.

### Cited Findings
- Burn 0.20 released 15 Jan 2026; ONNX importer overhauled (node-centric, burnpack format, mmap zero-copy weight loading); introduced CubeK kernels written in CubeCL — [Burn 0.20 release blog](https://burn.dev/blog/release-0.20.0/); [Phoronix](https://www.phoronix.com/news/Burn-0.20-Released)
- CubeCL targets NVIDIA CUDA, AMD ROCm HIP, Apple Metal, WebGPU, Vulkan, plus CPU with SIMD — [Burn 0.20 release blog](https://burn.dev/blog/release-0.20.0/)
- burn-onnx moved to its own repo; 0.21.0 (Mar 2026) adds an official-test compliance gate, onnx-ir graph simplification, opset 1–24 support for supported ops, validates 26–27 real-world models (classification, detection, depth, NLP, speech, generative); converts ONNX to generated Rust code + `.burnpack` weights at build time; unsupported ops via custom hooks — [tracel-ai/burn-onnx](https://github.com/tracel-ai/burn-onnx); [lib.rs burn-onnx](https://lib.rs/crates/burn-onnx)
- burn-onnx README does not mention face models/GANs or any performance comparison with ONNX Runtime — [tracel-ai/burn-onnx](https://github.com/tracel-ai/burn-onnx)
- candle-onnx: `simple_eval` is hardcoded to `Device::Cpu`; on InsightFace **det_10g.onnx** it took ~30 s/inference on CPU and ~2 s with a CUDA patch (issue open as of Apr 2026) — [huggingface/candle#3491](https://github.com/huggingface/candle/issues/3491)
- candle-onnx has missing ops (e.g. `Sign`) and quantized-ONNX eval errors; an OpsRegistry refactor was proposed — [candle#2638](https://github.com/huggingface/candle/issues/2638); [candle#1673](https://github.com/huggingface/candle/issues/1673)
- tract: CPU-only ONNX/TF inference, historically passing ~85% of ONNX backend tests and all real-life integration tests; used as an ort alternative backend (ort-tract 0.23) — [sonos/tract](https://github.com/kali/tract); [ort rc.13 notes](https://newreleases.io/project/github/pykeio/ort/release/v2.0.0-rc.13)
- wonnx (WebGPU ONNX runtime in Rust) was archived by its owner on 7 May 2025 — [webonnx/wonnx](https://github.com/webonnx/wonnx)
- cudarc exposes the CUDA driver API safely, including `ExternalMemory` via `CudaContext::import_external_memory()` (map whole or range; one mapped buffer at a time) — [docs.rs cudarc ExternalMemory](https://docs.rs/cudarc/latest/i686-pc-windows-msvc/cudarc/driver/safe/struct.ExternalMemory.html); [docs.rs cudarc](https://docs.rs/cudarc)
- A Rust crate `rust_insightface` exists for InsightFace detection and swap models (built on ort) — [xclud/rust_insightface](https://github.com/xclud/rust_insightface)

### Inferences
- Recommended: `ort` for all model inference; `cudarc` for custom CUDA kernels (pre/post-processing) on NVIDIA sharing ORT's CUDA stream/device buffers; optionally CubeCL for a single-source kernel set across CUDA/ROCm/Metal/Vulkan/WGPU.
- Burn could be evaluated for inswapper (it is essentially convs + AdaIN-style style blocks + Gemm), but numerical accuracy and speed vs TensorRT FP16 would need to be measured; treat as experimental.
- candle-onnx is not viable for real-time (seconds per frame on det_10g).

### Gaps
- No public benchmark of burn/candle/tract vs ORT on SCRFD/inswapper/GFPGAN found.
- No maintained Rust TensorRT binding (tensorrt-rs etc.) was found/verified in this session; TensorRT is best reached through ORT's TensorRT / TensorRT-RTX EPs.

## 4. wgpu / GPU pre- and post-processing; zero-copy interop

### Takeaway
Pre/post-processing (affine warp to 128×128, NCHW float normalization, mask feathering, paste-back, color transfer) should run on the GPU, but **wgpu has no portable external-memory import API**, so true zero-copy between wgpu and ORT's CUDA buffers is not available out of the box. The pragmatic zero-copy route on NVIDIA is to do pre/post in CUDA (cudarc/CubeCL) on the same device buffers ORT uses via IoBinding; wgpu is best for display/compositing and for non-NVIDIA backends.

### Cited Findings
- wgpu currently lacks a portable external-memory import API; tracking issue #2320 covers importing texture memory from fds (Linux), HANDLEs (Windows), AHardwareBuffer (Android) — [gfx-rs/wgpu#2320](https://github.com/gfx-rs/wgpu/issues/2320)
- Third-party crates provide platform-specific zero-copy imports into wgpu: `wgpu-external-frame` (Linux DMA-BUF via `VK_EXT_external_memory_dma_buf`, macOS IOSurface, Windows shared handles), `wgpu-native-texture-interop`/`wgpu-graft` (GL FBO, Vulkan image, Metal IOSurface) — [water-rs/wgpu-external-frame](https://github.com/water-rs/wgpu-external-frame); [docs.rs wgpu_native_texture_interop](https://docs.rs/wgpu-native-texture-interop/latest/wgpu_native_texture_interop/); [merely-made/wgpu-graft](https://github.com/merely-made/wgpu-graft)
- CUDA side: `cuImportExternalMemory` imports Vulkan-exported memory; coherence requires Vulkan barriers/semaphores (and flush/invalidate when CPU-mapped) — [CUDA Driver API External Resource Interop](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html); cudarc wraps this as `ExternalMemory` — [docs.rs cudarc](https://docs.rs/cudarc/latest/i686-pc-windows-msvc/cudarc/driver/safe/struct.ExternalMemory.html)
- A Vulkan↔CUDA interop walkthrough exists (export with `VK_KHR_external_memory_fd/win32`, import in CUDA) — [Medium: Vulkan-CUDA memory interoperability](https://medium.com/@mikolaj.gucki/vulkan-cuda-memory-interoperability-5442f3b43c3d)
- CubeCL compiles Rust kernels to CUDA, HIP, Metal, Vulkan (SPIR-V), WGSL — [Burn 0.20 blog](https://burn.dev/blog/release-0.20.0/)
- ONNX Runtime CUDA EP supports a **user compute stream** (requires IOBinding to avoid copies) — lets custom kernels and ORT run on the same stream without host sync — [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

### Inferences
- Zero-copy wgpu↔CUDA is possible in principle by allocating an exportable `VkDeviceMemory` via wgpu-hal's Vulkan backend (unsafe `as_hal`), exporting an fd/handle, importing via cudarc `import_external_memory`, and wrapping as an ORT tensor from a raw device pointer — but this is hand-built, Vulkan-only, needs external semaphores for sync, and is undocumented in any crate found.
- Simpler per-vendor plan: NVIDIA → CUDA kernels (cudarc/CubeCL) + ORT CUDA/TensorRT IoBinding (fully on-device); Apple → CoreML EP with CPU-side staging (unified memory makes copies cheap); AMD/Intel Windows → DirectML with CPU staging (or D3D12 resources if ort exposes DML device tensors); wgpu for final display.
- Crop tensors are tiny (1×3×128×128 f32 ≈ 196 KB; 512×512 enhancer input ≈ 3 MB) — the expensive transfers are full frames (1080p RGB ≈ 6 MB); keeping the full frame on GPU for paste-back matters more than zero-copy for the crops.

### Gaps
- No existing crate implementing wgpu→CUDA→ORT zero-copy was found.
- No benchmark numbers for wgpu compute warp/blend vs CPU for this workload found.

## 5. CPU image ops: fast_image_resize, image, yuvutils-rs, OpenCV

### Takeaway
On CPU, `fast_image_resize` (SIMD: SSE4.1/AVX2/NEON/WASM SIMD128) is ~20× faster than the `image` crate for bilinear resize and faster than libvips; use it (plus `yuvutils-rs` for camera YUV→RGB) instead of `image` on the hot path. OpenCV Rust bindings give warpAffine/seamlessClone parity with Python pipelines but add a heavy C++ build dependency.

### Cited Findings
- fast_image_resize 6.0.1 RGB8 4928×3279→852×567 on Ryzen 9 5950X: Bilinear — image 80.24 ms, resize 48.57 ms, libvips 5.58 ms, fir AVX2 **3.55 ms**; Lanczos3 — image 176.09 ms, libvips 17.31 ms, fir AVX2 **13.52 ms**; Nearest — image 30.36 ms vs fir 0.28 ms — [fast_image_resize x86_64 benchmarks](https://github.com/Cykooz/fast_image_resize/blob/main/benchmarks-x86_64.md)
- fast_image_resize supports SSE4.1, AVX2, NEON, WASM SIMD128; ARM64 benchmarks also published — [fast_image_resize ARM64 benchmarks](https://github.com/Cykooz/fast_image_resize/blob/main/benchmarks-arm64.md); [docs.rs](https://docs.rs/fast_image_resize)
- kornia-rs is a Rust low-level CV library (paper May 2025) positioning itself as an OpenCV alternative — [arXiv 2505.12425](https://arxiv.org/pdf/2505.12425)
- Deep-Live-Cam uses Poisson blending with feathering for seamless paste-back — [Deep-Live-Cam releases](https://github.com/hacksider/Deep-Live-Cam/releases)

### Inferences
- The affine warp (5-point alignment → 128×128 crop) and inverse warp-paste are not resize operations; implement them as GPU kernels or a SIMD bilinear sampler (kornia-rs has warp_affine; OpenCV crate has warpAffine). Poisson blending (seamlessClone) is expensive on CPU; a feathered alpha mask blend on GPU is the usual real-time substitute.
- Camera frames often arrive as NV12/YUY2/MJPEG; do YUV→RGB with yuvutils-rs (SIMD) or on GPU; avoid `image` crate conversions in the loop.

### Gaps
- yuvutils-rs benchmark numbers not retrieved in this session.
- No quantitative comparison of opencv-rust vs pure-Rust crates found; pros/cons (build complexity, CUDA-enabled OpenCV availability) are from general knowledge and should be verified.

## 6. Performance practices from FaceFusion / Deep-Live-Cam / VisoMaster and ONNX Runtime docs

### Takeaway
The big wins, in order: (1) TensorRT (or TensorRT-RTX) with engine + timing cache and FP16 (Deep-Live-Cam and VisoMaster report ~2–3× over CUDA EP); (2) keep enhancer optional/lower-rate (GFPGAN roughly halves FPS); (3) FP16 models (inswapper_128_fp16 is FaceFusion's default); (4) async/decoupled detection, batched multi-face inference, IoBinding + CUDA graphs to cut launch overhead; (5) Rust-side lock-free, zero-alloc hot path with stale-frame dropping.

### Cited Findings
- TensorRT EP options: `trt_engine_cache_enable` + `trt_engine_cache_path` cut session creation from minutes to seconds; `trt_timing_cache_enable` reduces build time (example 34.6 s → 7.7 s for efficientnet-lite4); `trt_fp16_enable`, `trt_int8_enable`; `trt_cuda_graph_enable`; `trt_builder_optimization_level` 0–5 (below 3 trades speed for build time); `trt_profile_min/opt/max_shapes` for dynamic dims; EP-context `_ctx.onnx` embedded engines give ~1.9 s init — [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- Cached TensorRT engines must be rebuilt when model, ORT/TensorRT version, or GPU changes — [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- CUDA EP `enable_cuda_graph`: removes per-kernel CPU launch overhead; requires static shapes, IOBinding, and all nodes on CUDA; first run captures — [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- CUDA EP tuning: `cudnn_conv_algo_search=EXHAUSTIVE` (recommended for fp16 to hit tensor cores), `cudnn_conv_use_max_workspace=1`, `prefer_nhwc` for tensor-core efficiency, `use_tf32` on by default on Ampere+, keep `do_copy_in_default_stream=true`, user compute stream with IOBinding; ORT 1.27+ ships CUDA 13.0/cuDNN 9 by default, CUDA 12.8 as separate packages — [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- CoreML EP: use `ModelFormat=MLProgram` (more ops; macOS 12+), `MLComputeUnits` (ALL / CPUAndGPU / CPUAndNeuralEngine), `RequireStaticInputShapes=1`, `ModelCacheDirectory` to cache compiled models, `AllowLowPrecisionAccumulationOnGPU` for fp16 accumulation, `SpecializationStrategy=FastPrediction`; dynamic shapes degrade performance — [ORT CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- Deep-Live-Cam 2.7 (Aug–Sep 2026): "Integrated TensorRT acceleration for NVIDIA GPUs (up to 3× faster, no quality loss)", face enhancer rebuilt on ONNX, PyTorch/TensorFlow/InsightFace/GFPGAN python deps removed, **batched multi-face inference**, **async face detection** to avoid camera stalls, "27fps with face enhancer on" (2.7 beta; GPU not stated) — [Deep-Live-Cam releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- VisoMaster runs models through TensorRT engines by default; claims 2–3× faster processing, lower VRAM, smoother real-time preview (vendor/secondary claims) — [visomaster.com](https://visomaster.com/); [DeepWiki VisoMaster advanced features](https://deepwiki.com/visomaster/VisoMaster/9-advanced-features)
- FaceFusion made `inswapper_128_fp16` the default swapper model — [FaceFusion changelog](https://docs.facefusion.io/introduction/changelog); fp16 inswapper checkpoints are distributed on Hugging Face — [hacksider/deep-live-cam inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx)
- Converting inswapper ONNX to TensorRT has reported conversion difficulties (Jetson) — [NVIDIA forum](https://forums.developer.nvidia.com/t/inswapper-onnx-model-conversion-to-tensorrt-model/310715)
- FP16 is slower than FP32 on GPUs/devices without native FP16 (software emulation) — [The Neural Base: FP16/INT8](https://theneuralbase.com/onnx/learn/intermediate/precision-fp16-int8/) (secondary tutorial source)
- Python overhead: ORT Python `run_with_iobinding` did not release the GIL (v1.11), blocking other threads during inference — [onnxruntime#11246](https://github.com/microsoft/onnxruntime/issues/11246); a practitioner writeup reports going from 10 FPS to 29 FPS by moving a video pipeline to zero-copy C++ ONNX (single anecdote) — [HackerNoon](https://hackernoon.com/python-is-a-video-latency-suicide-note-how-i-hit-29-fps-with-zero-copy-c-onnx)
- A pure-Rust real-time face-swap app (RetinaFace + ArcFace + InSwapper + optional GFPGAN, all via `ort`, egui UI, nokhwa capture) reports 60 FPS on webcam using 3 threads with "zero locks on hot path" (ArcSwap + atomic generation counters), cache-line-aligned shared state, zero allocations on hot path except one Arc per frame; GPU/per-stage timings not disclosed; author asks about multi-session ORT parallelism — [Habr article 1024700](https://habr.com/ru/articles/1024700/) (date unclear from fetch; treat 60 FPS claim as unverified)

### Inferences
- Pipeline design for Rust: capture thread → bounded channel (capacity 1–2, `try_send` + drop-oldest, or `ArcSwap`/triple buffer "latest frame wins") → inference thread → render thread. Dropping stale frames bounds latency to ~1 frame.
- Run detection (SCRFD 640×640) every N frames or asynchronously and track boxes/landmarks in between (Deep-Live-Cam's "async face detection" is the same idea); compute the ArcFace source embedding once per source face (and apply inswapper's emap projection once).
- Batch all faces in a frame into one inswapper call (N×3×128×128) — requires dynamic batch dim in TensorRT profiles (`trt_profile_*_shapes`), or pad to a fixed max batch to keep CUDA graphs usable.
- Pre-create all sessions at startup with engine/timing caches; warm up with dummy inputs to trigger cuDNN algo search / CUDA graph capture before the first real frame.
- INT8 for inswapper/GFPGAN: no published quality evaluation found; GAN-style generators are typically sensitive to INT8 — recommend FP16 as the default and treat INT8 as experimental.

### Gaps
- No rigorous Python-vs-Rust benchmark of the same inswapper pipeline was found.
- No official FaceFusion TensorRT-vs-CUDA numbers found (FaceFusion's GPU page only lists TFLOPS tables — [FaceFusion GPU capabilities](https://docs.facefusion.io/knowledgebase/gpu-capabilities)).
- No INT8 quality study for inswapper/GFPGAN found.

## 7. Realistic FPS numbers (inswapper_128 + SCRFD) on RTX 3060/4060/4090, Apple M-series, CPU

### Takeaway
Reliable, reproducible FPS benchmarks are scarce; available numbers come mostly from project release notes and secondary blog aggregators: roughly **~20+ FPS on RTX 4060 (enhancer off)**, **60+ FPS at 1080p on RTX 4090**, enhancer (GFPGAN) roughly **halves** FPS, TensorRT adds up to ~3×. CPU-only is below real time. No trustworthy Apple M-series CoreML numbers found.

### Cited Findings
- RTX 4060 8 GB ≈ 20+ FPS; RTX 4090 60+ FPS at 1080p; CPU-only/older GPUs cannot sustain 24+ FPS without dropping to 720p or lower; disabling GFPGAN roughly doubles FPS — [Starlog article on Deep-Live-Cam](https://starlog.is/articles/ai-dev-tools/hacksider-deep-live-cam/) (secondary aggregator; treat as indicative only)
- Deep-Live-Cam 2.7 beta: 27 FPS with face enhancer on (hardware unspecified); TensorRT "up to 3× faster" — [Deep-Live-Cam releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- FaceFusion docs advise frame enhancement is not recommended on non-RTX 4090 cards due to performance cost — [LooPIN FaceFusion usage guide](https://docs.loopin.network/tutorials/faceswap/facefusion-usage) (third-party guide)
- Rust ort-based app claims 60 FPS webcam (hardware not disclosed) — [Habr](https://habr.com/ru/articles/1024700/)

### Inferences
- Rough per-frame budget for 30 FPS is 33 ms; at 60 FPS, 16.7 ms. inswapper_128 is small per call; GFPGAN (512×512) is the dominant cost, so running the enhancer at reduced rate or only on the swapped crop at lower resolution (e.g. GPEN-256) is the main lever on mid-range GPUs.
- Expect Apple M-series with CoreML (MLProgram, fp16) to be viable for swap-only real-time, but numbers must be measured.

### Gaps
- No primary-source FPS benchmarks for RTX 3060, Apple M1–M4 with CoreML, or modern CPUs (e.g. with OpenVINO/XNNPACK) for inswapper_128 + SCRFD were found; the report should present FPS as ranges from secondary sources and recommend in-app benchmarking.
