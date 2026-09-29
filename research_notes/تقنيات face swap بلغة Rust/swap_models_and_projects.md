# Real-time Face Swap Models and Existing Projects (state as of Sept 2026)

Research date: 2026-09-28. Primary sources are GitHub repos/source code (read directly via GitHub API/raw files), HuggingFace, arXiv, and project docs. Star counts and dates are from the GitHub API on 2026-09-28.

## Q1. Which swap models exist and are practical for real-time? (resolution, speed, ONNX, embedding input)

### Takeaway
The practical real-time swappers in 2026 are still all small one-shot GAN-style encoder/decoder networks that run in ONNX: inswapper_128 (+fp16), HyperSwap 1a/1b/1c 256, AlphaFace 256 (new, Jan 2026), GHOST 1/2/3 256, SimSwap 256/512, HiFiFace-unofficial 256, BlendSwap 256, UniFace 256, ReSwapper (128/256). FaceFusion 3.9 ships 14 of these as ONNX files. Almost all use a 512-d ArcFace embedding (usually InsightFace w600k_r50), sometimes passed through a per-model projection (inswapper "emap", AlphaFace "emp.npy", FaceFusion "CrossFace" converters). Diffusion swappers (DiffSwap, REFace, DiffFace, DreamID, VFace) are still 0.6 to 26 s per image and not real-time. InsightFace's newer inswapper-512-live is real but is **not publicly downloadable**.

### Cited Findings

**FaceFusion's model catalog (the best single reference for ONNX-ready swappers)**
- FaceFusion 3.9.0's face_swapper choices, with the "pixel boost" output sizes each supports: alphaface_256, blendswap_256, ghost_1_256, ghost_2_256, ghost_3_256, hififace_unofficial_256, hyperswap_1a_256, hyperswap_1b_256, hyperswap_1c_256, inswapper_128, inswapper_128_fp16, simswap_256, simswap_unofficial_512, uniface_256. inswapper supports 128 up to 1024. The 256 models support 256 up to 1024. simswap_unofficial_512 supports 512 up to 1024. — [facefusion choices.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/choices.py)
- Per-model metadata in FaceFusion core.py. Format: type / alignment template / input size / normalization / vendor / license label.
  - inswapper_128 and inswapper_128_fp16: arcface_128 template, 128×128, mean 0 / std 1, vendor InsightFace, "Non-Commercial". Hosted in models-3.0.0.
  - hyperswap_1a/1b/1c_256: arcface_128 template, 256×256, mean/std 0.5, vendor FaceFusion, "ResearchRAIL". Hosted in models-3.3.0.
  - alphaface_256: arcface_128 template, 256×256, mean 0 / std 1, vendor AlphaFace, "Non-Commercial". Added in models-3.9.0.
  - ghost_1/2/3_256: arcface_112_v1 template, 256, mean/std 0.5, vendor ai-forever, "Apache-2.0". Needs an extra `crossface_ghost.onnx` embedding converter (models-3.4.0).
  - hififace_unofficial_256: mtcnn_512 template, 256, vendor GuijiAI, license "Unknown". Needs `crossface_hififace.onnx`.
  - simswap_256 and simswap_unofficial_512: arcface_112_v1 template, vendor neuralchen, "Non-Commercial". Needs `crossface_simswap.onnx`.
  - blendswap_256: ffhq_512 template, vendor mapooon, "Non-Commercial".
  - uniface_256: ffhq_512 template, vendor xc-csc101, license "Unknown".
  - Source: [facefusion face_swapper/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
- How each model takes its source identity, from FaceFusion `prepare_source_embedding` and `forward_swap_face`:
  - **blendswap and uniface** take an aligned *source image* as input, not an embedding.
  - **inswapper** takes the ArcFace embedding multiplied by the "emap" matrix. The emap is read from the ONNX model's last initializer. The result is then divided by the embedding's norm.
  - **hyperswap** takes the L2-normalized ArcFace embedding (`embedding_norm`).
  - **alphaface** takes the raw embedding.
  - **ghost, simswap and hififace** first pass the w600k embedding through a "CrossFace" converter ONNX model.
  - A "face_swapper_weight" option mixes the target's embedding into the source's by up to ±0.35.
  - Source: [facefusion face_swapper/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
- FaceFusion 3.9.0 (2026-09-03) release notes: "Introduce `alphaface_256` model to the face swapper" and a new `hrffa` face landmarker. — [FaceFusion releases](https://github.com/facefusion/facefusion/releases)

**inswapper_128 (InsightFace)**
- Inputs:
  - target: a [1,3,128,128] RGB aligned crop in the range [0,1].
  - source: a [1,512] latent. It is the ArcFace IResNet50 embedding (input 112×112, normalized to [-1,1]), L2-normalized, multiplied by `emap`, then renormalized.
  - "emap" can be extracted from the inswapper ONNX file.
  - Source: [ReSwapper README](https://github.com/somanchiu/ReSwapper)
- InsightFace's own figures: INSwapper_128 has 138.3M params and 174.7 GFLOPs. Scores: Similarity 86.9, Realism 63.3, Attributes 78.8. — [deepinsight/inswapper-512-live](https://github.com/deepinsight/inswapper-512-live)
- **"inswapper 512" is not a rumour, but it is not public.** InsightFace's inswapper-512-live claims:
  - native 512×512 output
  - fewer than 1/10 of inswapper_128's compute
  - base: 9.5M params, 12.1 GFLOPs, Similarity 87.0, Realism 73.7, Attributes 80.1
  - mini: 4.7M params, 6.3 GFLOPs
  - It ships only inside the Picsi.ai macOS app (2025-11-18) and iOS app (2025-08-18).
  - License: "strictly for academic use or personal testing and cannot be used for any commercial purposes."
  - Newer closed variants are INSwapper_Cyn and INSwapper_Dax (Dax with optimizer: Similarity 93.4, Realism 90.2).
  - Source: [deepinsight/inswapper-512-live](https://github.com/deepinsight/inswapper-512-live)
- InsightFace README: "For inswapper series face swap models (e.g., inswapper_128.onnx/inswapper-512-live), please contact contact@insightface.ai for licensing." InsightFace 1.0 (2026-05-23) added a desktop GUI demo that includes face-swap trials. — [insightface README](https://github.com/deepinsight/insightface)

**HyperSwap (FaceFusion Labs)**
- HyperSwap 1a/1b/1c at 256 px have been available since FaceFusion 3.3.0. They are licensed ResearchRAIL(-MS); commercial use requires contacting FaceFusion Labs. — [FaceFusion licenses docs](https://docs.facefusion.io/introduction/licenses); [facefusion-labs/hyperswap](https://github.com/facefusion/facefusion-labs/tree/master/hyperswap); [HF hyperswap_1a_256.onnx](https://huggingface.co/facefusion/models-3.3.0/blob/main/hyperswap_1a_256.onnx)
- Deep-Live-Cam's paid 2.7 build lists "Inswapper and Hyperswap" as selectable models, alongside ReSwapper. — [Deep-Live-Cam 2.7-RC1 release](https://github.com/hacksider/Deep-Live-Cam/releases)

**AlphaFace (new, 2026)**
- The paper (arXiv 2601.16429, 2026-01-23, Yu et al.) claims better robustness to extreme poses "while maintaining real-time performance". It uses a vision-LLM plus CLIP semantic contrastive losses at training time only, with no extra inference cost. The abstract gives no FPS figure. — [arXiv 2601.16429](https://arxiv.org/abs/2601.16429)
- The official repo reports license MIT (GitHub API; 73 stars). Weights are distributed via Google Drive with no separate weights license stated. — [andrewyu90/Alphaface_Official](https://github.com/andrewyu90/Alphaface_Official)
- **The license signals conflict:**
  - The repo says MIT.
  - The arXiv paper is CC BY-NC-SA 4.0 ([arXiv](https://arxiv.org/abs/2601.16429)).
  - FaceFusion labels the model "Non-Commercial" ([FaceFusion licenses](https://docs.facefusion.io/introduction/licenses)).
- VisoMaster-Fusion's integration notes:
  - The model is 256×256, a single `alphaface_swapper_fused_norm.onnx` of about 529 MiB (FP32).
  - It runs as FP16 under TensorRT.
  - It reuses the W600K ArcFace embedding, projected by a 512×512 matrix `emp.npy` from the authors.
  - It uses the pose-aware "arcfacemap" alignment templates (shared with GhostFace).
  - The exported graph needed ONNX Runtime symbolic shape inference before TensorRT would accept it, because of dynamic reflect-pad Shape/Gather ops.
  - Instance norm was fused into a single InstanceNormalization node, and the AdaIN step was simplified.
  - Source: [VisoMaster-Fusion docs/alphaface.md](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/alphaface.md)

**ReSwapper (community reproduction of inswapper)**
- Repo license AGPL-3.0; 237 stars; last push 2025-06-14. — [GitHub API / somanchiu/ReSwapper](https://github.com/somanchiu/ReSwapper)
- It reproduces the inswapper architecture ("InSwapper and SimSwap are extremely similar").
- Its supervised training mode uses FFHQ images as targets and **inswapper outputs as the ground-truth swapped images**, so the weights are effectively distilled from inswapper.
- It provides 128 and 256 checkpoints (trained at multiple resolutions) and exports to ONNX usable with the original INSwapper class.
- It documents an alignment offset fix for non-128 resolutions: Offset ≈ (128/32768)·Resolution − 0.5.
- Training ran on an RTX 3060: about 12 h per 50k steps.
- Source: [ReSwapper README](https://github.com/somanchiu/ReSwapper)

**GHOST (sber-swap)**
- Repo license Apache-2.0; last release sber-swap-v2.0 in Jan 2022 (old). — [ai-forever/ghost](https://github.com/ai-forever/ghost)
- FaceFusion labels the GHOST weights Apache-2.0. The CrossFace converter it needs is labelled OpenRAIL-M. — [FaceFusion licenses](https://docs.facefusion.io/introduction/licenses)
- GHOST 2.0 (arXiv 2502.18417, 2025) is about head swapping, not a real-time face swapper. — [arXiv 2502.18417](https://arxiv.org/pdf/2502.18417)

**SimSwap / SimSwap-512**
- "For academic and non-commercial use only… CC-BY-NC 4.0."
- The 512 model is a 2021 beta trained on VGGFace2-HQ. Last push 2024.
- Source: [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap)

**VisoMaster-Fusion's model list**
- Swappers: Inswapper128 (internal resolution 128/256/384/512 with Auto Resolution), AlphaFace, InStyleSwapper256 A/B/C, SimSwap512, GhostFace v1/v2/v3, CSCS, and DeepFaceLive DFM models.
- InStyleSwapper256 are 256 px swappers "based on the Inswapper architecture and trained using a custom technique". They have an optional 512 "sub-pixel phase-shift tiling" mode.
- CSCS "combines two embeddings (appearance + identity)".
- Source: [VisoMaster-Fusion user manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)

**Research papers from 2025–2026**
- **CanonSwap** (arXiv 2507.02691, ICCV 2025): video face swapping. It separates motion from appearance by warping to a canonical space, then injects identity through a "Partial Identity Modulation" module with a spatial mask. It targets temporal consistency, not explicitly real-time. — [arXiv 2507.02691](https://arxiv.org/abs/2507.02691)
- **AlphaFace** (Jan 2026): real-time and pose-robust (see above). — [arXiv 2601.16429](https://arxiv.org/abs/2601.16429)
- **Diffusion swappers are not real-time.** Measured per image on an A100:

  | Method | Seconds per image |
  |---|---|
  | DiffFace | 25.8 |
  | DiffSwap | 7.82 |
  | FaceAdapter | 3.42 |
  | REFace | 3.75 |
  | DreamID (SIGGRAPH Asia 2025, SD-Turbo based, 512×512) | 0.6 |

  Source: [DreamID arXiv 2504.14509](https://arxiv.org/html/2504.14509v3)
- Other 2025–2026 diffusion or video items found:
  - DiffSwap++ (arXiv 2511.05575) — [arXiv](https://arxiv.org/pdf/2511.05575)
  - VFace, training-free diffusion video swapping (arXiv 2602.07835) — [arXiv](https://arxiv.org/pdf/2602.07835)
  - HiFiVFS (arXiv 2411.18293) — [arXiv](https://arxiv.org/pdf/2411.18293)
  - GSwap, Gaussian head swap (arXiv 2603.23168) — [arXiv](https://arxiv.org/pdf/2603.23168)
  - Survey and benchmark "Towards High Fidelity Face Swapping" (arXiv 2605.00883, 2026-04-27), which introduces the CASIA FaceSwapping benchmark and organizes methods into five paradigms — [arXiv 2605.00883](https://arxiv.org/abs/2605.00883)
- Deep-Live-Cam's "FLUX Live" prompt-editing mode reaches 30 fps only on RTX 5090 / RTX 6000-class GPUs. Its "Decart Live" mode is cloud diffusion (Lucy 2.0/2.1) and needs an API key. These are the closest things to real-time diffusion in the ecosystem. — [Deep-Live-Cam 2.7-RC1 release](https://github.com/hacksider/Deep-Live-Cam/releases)

### Inferences
- A Rust app can support every FaceFusion swapper through one generic ONNX interface: an aligned crop at size S, plus either (a) a 512-d embedding with a per-model transform (emap, normalization, CrossFace converter, or a projection matrix) or (b) an aligned source image. The alignment template matters per model: arcface_128, arcface_112_v1, ffhq_512, mtcnn_512, and arcfacemap for AlphaFace/Ghost.
- In practice, HyperSwap and AlphaFace at 256 are the 2026 quality upgrade over inswapper_128. inswapper-512-live shows that a 512-output real-time model can be tiny (6–12 GFLOPs versus 175), but it cannot be obtained.
- Because of the inswapper-512-live compute figures, a from-scratch small 256/512 swapper is a credible long-term goal for a permissively licensed project.

### Gaps
- I found no reliable, like-for-like FPS or latency benchmarks per model on common GPUs (e.g., RTX 3060/4090) for HyperSwap, AlphaFace, GHOST, SimSwap or UniFace. Treat project claims as unverified: Deep-Live-Cam "27fps with face enhancer on" (2.7-beta) and despite-death/face-swap "60 fps".
- HyperSwap parameter count and FLOPs, and the facefusion-labs training-code license, were not verified.
- UniFace and BlendSwap origins and papers were not investigated beyond FaceFusion's vendor labels.
- Whether REFace or DiffSwap have ONNX exports was not checked; they are irrelevant for real-time anyway.
- I found no evidence of an official "inswapper_256".

## Q2. Licenses: which models are genuinely permissive and bundleable?

### Takeaway
Almost nothing in the swap chain is cleanly permissive. inswapper (all versions), SimSwap, BlendSwap, AlphaFace (per FaceFusion) and InsightFace's ArcFace/SCRFD/RetinaFace are non-commercial. HyperSwap is ResearchRAIL. HiFiFace-unofficial and UniFace are unknown. ReSwapper is AGPL code whose weights are distilled from inswapper outputs. GHOST (Apache-2.0) is the only swapper with a permissive weights label, but it still needs an identity encoder, and the default w600k ArcFace is non-commercial. An open-source app can therefore permissively bundle only non-swap models (YuNet, BiSeNet, 2DFAN4, GFPGAN, etc.) and should download swap models on the user's request with their licenses shown.

### Cited Findings
- FaceFusion's own license table:
  - **Permissive:** GHOST Apache 2.0; GFPGAN Apache 2.0; BiSeNet MIT; YuNet MIT; 2DFAN4 MIT; HRFFA MIT; Live Portrait MIT; RestoreFormer Apache 2.0; Real ESRGAN BSD-3.
  - **Non-commercial:** INSwapper; ArcFace; SimSwap; BlendSwap; AlphaFace; SCRFD; RetinaFace; GPEN.
  - **Other:** HyperSwap ResearchRAIL; CrossFace OpenRAIL-M; FAN 68/5 OpenRAIL-M; CodeFormer S-Lab 1.0; XSeg GPL-3.0; YOLO Face GPL-3.0; HifiFace Unknown; UniFace None.
  - Source: [FaceFusion licenses](https://docs.facefusion.io/introduction/licenses)
- InsightFace: "The code of InsightFace is released under the MIT License… The training data… (and the models trained with these data) are available for non-commercial research purposes only." Inswapper models need licensing from contact@insightface.ai. — [insightface README](https://github.com/deepinsight/insightface)
- inswapper-512-live: "strictly for academic use or personal testing and cannot be used for any commercial purposes." — [inswapper-512-live](https://github.com/deepinsight/inswapper-512-live)
- SimSwap: CC-BY-NC 4.0. — [SimSwap](https://github.com/neuralchen/SimSwap)
- GHOST repo: Apache-2.0. — [ai-forever/ghost](https://github.com/ai-forever/ghost)
- ReSwapper repo: AGPL-3.0. Its training uses inswapper outputs as targets. — [ReSwapper](https://github.com/somanchiu/ReSwapper)
- AlphaFace: repo MIT, paper CC BY-NC-SA 4.0, and FaceFusion labels it Non-Commercial (conflicting). — [Alphaface_Official](https://github.com/andrewyu90/Alphaface_Official); [arXiv](https://arxiv.org/abs/2601.16429); [FaceFusion licenses](https://docs.facefusion.io/introduction/licenses)
- XSeg occluder models (DeepFaceLab) are GPL-3.0. BiSeNet ResNet18/34 face parsers (vendor yakhyo) are MIT. — [facefusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)

### Inferences
- **Recommended Rust design:** keep the code permissive (MIT/Apache), ship no swap weights, and use a model manifest that downloads from the original hosts with SHA-256 pinning. Show each model's license in the UI. noperson (Rust) and FaceFusion already pin by hash.
- **Most permissive stack reachable today:**
  - detection: YuNet (MIT)
  - landmarks: 2DFAN4 / HRFFA (MIT)
  - parsing mask: BiSeNet (MIT)
  - swapper: GHOST (Apache-2.0), with its embedding supplied via CrossFace (OpenRAIL-M, which carries use restrictions)
  - enhancer: GFPGAN (Apache-2.0)
  - The identity-embedding model is the weak link: GHOST's own ArcFace backbone license was not verified.
- AlphaFace's MIT repo label may allow permissive use of its code and weights, but because of the conflicting labels you would need to ask the authors before bundling.

### Gaps
- The exact license text applying to GHOST's pretrained ArcFace backbone and weights files was not verified.
- The HyperSwap ResearchRAIL-MS full text was not fetched.
- The license of VisoMaster-Fusion's InStyleSwapper256 and CSCS weights was not verified.
- The license of AlphaFace's Google-Drive weights is not stated separately in the repo.

## Q3. Existing projects: state, features, license, and what each does well technically

### Takeaway
Deep-Live-Cam (AGPL-3.0, about 97k stars) is now an open-core product. The GitHub build is inswapper_128_fp16 + GFPGAN with notable low-level optimizations. The paid "Ultimate" 2.7.x adds TensorRT, HyperSwap/ReSwapper, GPEN, a PyQt6 UI, Flux/Decart modes and more. FaceFusion 3.9 (OpenRAIL-AS) is the most complete modular pipeline. VisoMaster-Fusion (GPL-3.0) is the most feature-rich for masks and restoration and has TensorRT FP16 engineering. Rope was revived as "Rope-Bronze" in July 2026. DeepFaceLive is archived. Several small Rust ports appeared in 2025–2026, none mature.

### Cited Findings

**Deep-Live-Cam** ([repo](https://github.com/hacksider/Deep-Live-Cam))
- AGPL-3.0; about 96.9k stars; pushed 2026-09-26. Latest release 2.7.5-Ultimate (2026-09-07). — GitHub API
- 2.7.5 added Face Fader (gradual morph), Favorites, Library and Face Picker. It is "Exclusively Available on our website". — [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- 2.7-ultimate (2026-08-01) claims:
  - a 35k-line rewrite
  - TensorRT "up to 3× faster"
  - PyTorch, TensorFlow, InsightFace and GFPGAN dependencies removed in favour of pure ONNX
  - batched multi-face inference and async face detection
  - Poisson + feathering blend, lip-mask fixes, color matching
  - real-time frame interpolation, the ReSwapper model, and a PyQt6 UI
  - an auto GPU installer (NVIDIA incl. RTX 50, AMD, Apple Silicon)
  - video export in its own process, and multi-GPU AMD DirectML
  - Source: [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- 2.7-beta (2026-03-11) added: real-time face enhancer, "Inswapper optimizer", GPEN 256/512, lip, chin and eyes masks, interpolation, LUTs, window projection, and a claim of "27fps with face enhancer on". 2.6 (2026-02) added virtual camera support and a rebuilt core. 2.4 (2025-12) improved Poisson blending and the mouthmask. — [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- The open GitHub README instructs downloading only `inswapper_128_fp16.onnx` and `gfpgan-1024.onnx`. It supports Python 3.11–3.14 and the cuda, coreml and other providers. — [README](https://github.com/hacksider/Deep-Live-Cam)
- Techniques in the open source code (`modules/processors/frame/face_swapper.py`, about 1,580 lines):
  - an ONNX Runtime CUDA session with `enable_cuda_graph` and IOBinding, guarded by a lock
  - a cached, feathered alpha template in aligned-face space for a fast paste-back that writes in place on the GPU path
  - Poisson blending via `cv2.seamlessClone`, using the warped paste mask with a cached fallback ellipse (comments explicitly avoid EMA/lag)
  - an OpenCV-CUDA GaussianBlur/sharpen/resize helper module (`gpu_processing.py`)
  - fp16 model auto-selection
  - an OpenVINO provider config
  - Source: [face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py); [gpu_processing.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/gpu_processing.py)
- `onnx_optimize.py` rewrites the inswapper graph for CoreML on Apple Silicon. It replaces `Pad(reflect)` ops that caused CPU fallbacks and folds Shape→Gather chains into constants for a fixed input shape, caching the result as `*_coreml*.onnx`. — [onnx_optimize.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/onnx_optimize.py)

**FaceFusion** ([repo](https://github.com/facefusion/facefusion))
- License OpenRAIL-AS ("Copyright (c) 2026 Henry Ruhs"); about 30k stars; latest 3.9.0 (2026-09-03). — [LICENSE.md](https://github.com/facefusion/facefusion/blob/master/LICENSE.md); GitHub API
- Processors: age_modifier, background_remover, deep_swapper (DeepFaceLive DFM models), expression_restorer, face_debugger, face_editor, face_enhancer, face_swapper, frame_colorizer, frame_enhancer, lip_syncer. — [repo tree](https://github.com/facefusion/facefusion/tree/master/facefusion/processors/modules)
- Execution providers: cuda, tensorrt, rocm, migraphx, coreml, openvino, qnn, directml, cpu (all ONNX Runtime). — [choices.py](https://github.com/facefusion/facefusion/blob/master/facefusion/choices.py)
- Model hub: versioned HuggingFace repos facefusion/models-3.0.0 through models-3.9.0, `deepfacelive-models-*` (iperov, mats, rumateus, druuzil, jen, edel), and `libraries-4.0.0`. Each model has a .hash file for verification. — [HF facefusion](https://huggingface.co/facefusion); [core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
- 3.8.0 (2026-07-31) added an FFmpeg-based video manager, memory versus disk `--workflow-strategy`, AV1, alpha channel and HDR handling. 3.8.1 fixed a VRAM leak in the inference pool, added cuda@12/13 installer options, and fixed fp16 under CoreML. — [releases](https://github.com/facefusion/facefusion/releases)
- The swap pipeline:
  1. Warp by 5 landmarks to the model template at the pixel-boost size.
  2. Build masks: box (blur and padding), occlusion (XSeg 1/2/3), area (from 68 landmarks), and region (BiSeNet parsing).
  3. Combine the masks with a per-pixel minimum.
  4. Run pixel-boost inference, then paste back.
  - Source: [core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)

**VisoMaster / VisoMaster-Fusion**
- The original VisoMaster (GPL-3.0) has been stale since v0.1.6 (2025-03). The community fork VisoMaster-Fusion (GPL-3.0; 924 stars) is active, with v3.12.3 released 2026-08-29. — GitHub API: [VisoMaster](https://github.com/visomaster/VisoMaster), [VisoMaster-Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion)
- Backends: CPU, CUDA, TensorRT, TensorRT-Engine. It recommends 6 GB or more of VRAM (8–12 GB with several models) and 20–30 GB of disk. The portable install uses PyTorch 2.8 + CUDA 12.9. — [user manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)
- Features:
  - swappers as listed in Q1, plus DFM models with "AMP Morph Factor" and RCT color transfer
  - masks: Occluder, XSeg (with Mouth/Lips Protection), FaceParser per region, Restore Eyes/Mouth, and "Mouth Fit & Align"
  - AutoColor Transfer and Ending Color Transfer
  - restorers: GFPGAN, CodeFormer, GPEN up to 2048, RestoreFormer++, VQFR
  - a Ref-LDM diffusion denoiser
  - LivePortrait-style Face Expression Restorer and face editor
  - EMA keypoint smoothing, a YuNet detector option, VR180 mode, live webcam and a job manager
  - Source: [user manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)

**Rope**
- Hillobar/Rope was revived with "Rope-Bronze" (2026-07-11). The GitHub API shows no license. — [Rope](https://github.com/Hillobar/Rope)
- Rope-Bronze adds:
  - TRT engine support
  - "Batched inswapper for better 256 and 512 mode performance"
  - LAB color matching
  - an XSeg masker
  - Likeness / Fidelity settings
  - a screen-capture mode that swaps the contents of a desktop window
  - Source: [Rope README](https://github.com/Hillobar/Rope)

**roop and DeepFaceLive**
- s0md3v/roop is archived (AGPL-3.0). C0untFloyd/roop-unleashed returns "Repository access blocked" on GitHub. — GitHub API: [roop](https://github.com/s0md3v/roop)
- iperov/DeepFaceLive is archived (GPL-3.0; about 31k stars; last release Jan 2023). It has two swap modes:
  - "Face Swap (DFM)": per-identity models trained with DeepFaceLab. This gives the highest quality, but you must train a model per person.
  - "Face Swap (Insight)": a one-shot mode.
  - It also has a Face Animator. It ships DirectX12 (NVIDIA/AMD/Intel) and NVIDIA builds, uses ONNX, and recommends an RTX 2070 / RX 5700 XT or better.
  - Source: [DeepFaceLive README](https://github.com/iperov/DeepFaceLive)

**Native (Rust / C++) reimplementations** (GitHub search, 2026-09-28)
- **despite-death/face-swap** (Rust, MIT, 9 stars, 2026-04):
  - Pipeline: RetinaFace → ArcFace w600k_r50 → inswapper_128_fp16 + emap.bin → optional GFPGAN → affine paste-back.
  - Architecture: `cam-capture thread → ArcSwap<Frame> → ml-pipeline thread → ArcSwap<Frame> → egui`, i.e. lock-free frame handoff with no heap allocation on the hot path.
  - It claims 60 fps.
  - Source: [despite-death/face-swap](https://github.com/despite-death/face-swap)
- **Jagirium/noperson** (Rust, AGPL-3.0, 2026-08):
  - "GPU-first": inference and image processing stay on the GPU across the live path, using ONNX Runtime CUDA with optional TensorRT.
  - Linux/NVIDIA only so far; V4L2 input with v4l2loopback output.
  - Supports user-supplied DFM models.
  - Every model is pinned by SHA-256 in `src/models/registry.rs`.
  - Source: [noperson](https://github.com/Jagirium/noperson)
- **LdDl/deep_faceswap** (Rust, a "port of Deep-Live-Cam", 2026-04):
  - YOLOv8n / det_10g, ArcFace, inswapper_128, GFPGAN, mouth mask, multi-face mapping, CUDA through ORT.
  - "ROI-based paste_back… ~50-100x faster on 4K, ~3.7x end-to-end on 720p video".
  - An actix-web REST API and a SvelteKit UI. No OpenCV dependency.
  - Source: [deep_faceswap](https://github.com/LdDl/deep_faceswap)
- **itsVentie/Saccade** (Rust, Tauri v2 + Preact, 2026-09): Tokio channels, zero-copy buffer pooling, and the `ort` crate with CUDA/DirectML/CoreML. Still in the roadmap phase. — [Saccade](https://github.com/itsVentie/Saccade)
- Other Rust repos:
  - xclud/rust_insightface: a crate for InsightFace detection and swap. — [rust_insightface](https://github.com/xclud/rust_insightface)
  - michelkluger/facenaissance: SCRFD + ArcFace + inswapper_128 in Rust, on-device. — [facenaissance](https://github.com/michelkluger/facenaissance)
  - Loner1024/swapper-rs (2025-03). — [swapper-rs](https://github.com/Loner1024/swapper-rs)
- C++ reimplementations:
  - CodingRookie98/FaceFusionCpp (GPL-3.0, active 2026-09) — [FaceFusionCpp](https://github.com/CodingRookie98/FaceFusionCpp)
  - hpc203/facefusion-onnxrun (C++/Python ORT, 2024) — [facefusion-onnxrun](https://github.com/hpc203/facefusion-onnxrun)
  - xlite-dev/lite.ai.toolkit (GPL-3.0, 4.4k stars; MNN/ORT/TRT including Face-Fusion) — [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit)
  - anzisheng/swap_face_tensorrt (MIT) — [swap_face_tensorrt](https://github.com/anzisheng/swap_face_tensorrt)

### Inferences
- The strongest open, reusable engineering ideas are:
  - Deep-Live-Cam's CUDA-graph + IOBinding swap session, cached feather template and CoreML graph surgery
  - FaceFusion's declarative model registry (template, size, normalization, hashes, license), wide EP matrix and composable masks
  - VisoMaster-Fusion's TensorRT FP16 allowlist, cached shape-inferred ONNX and cached TRT engines
  - Rope's batched pixel-boost sub-inferences
  - noperson's and despite-death's lock-free, GPU-resident Rust pipelines
- There is no mature Rust competitor. The existing ones are small (under 10 stars) and mostly limited to inswapper.
- Deep-Live-Cam's newest capabilities are closed/paid. The open build is one generation behind. That is a market gap an open, fast Rust app could fill.

### Gaps
- Rope-Next, FaceFusion Next, the current state of roop-unleashed (access blocked), and the content of FaceFusion's `libraries-4.0.0` HF repo (possibly FaceFusion 4) were not verified.
- Independent benchmarks of Deep-Live-Cam's "3× TensorRT" and "27 fps" claims were not found.
- I did not verify whether the paid Deep-Live-Cam binary's code is published under AGPL.

## Q4. Real-time quality tricks: occlusion masks, mouth mask, color transfer, pixel boost, temporal smoothing

### Takeaway
The common toolkit is:
- multiple masks combined by per-pixel minimum: box/feather, an XSeg occluder, BiSeNet region parsing, and landmark area masks
- mouth/lip restoration from the original frame
- color transfer (LAB/RCT) and Poisson blending
- "pixel boost", which in FaceFusion is sub-pixel interleaving (space-to-depth), not spatial tiling
- landmark EMA smoothing
- an optional GAN restorer (GFPGAN/GPEN) and expression restoration (LivePortrait-style)

### Cited Findings
- **FaceFusion pixel boost is sub-pixel interleave, not tiling.**
  - `implode_pixel_boost` reshapes an (N·S)×(N·S) crop into N² images of S×S, each a strided subsample: reshape(S,N,S,N,3) then transpose(1,3,0,2,4).
  - Each subsample is swapped separately, then `explode_pixel_boost` interleaves the results back.
  - So 512 boost on inswapper_128 means 16 inferences per face.
  - Source: [pixel_boost.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/pixel_boost.py); [core.py swap_face](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
- VisoMaster-Fusion calls the same idea "sub-pixel phase-shift tiling" at 512. It also offers an Auto Resolution mode that picks 128–512 based on detected face size. — [VisoMaster-Fusion manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)
- Rope-Bronze batches these sub-inferences ("Batched inswapper for better 256 and 512 mode performance"). — [Rope README](https://github.com/Hillobar/Rope)
- **Masks.**
  - FaceFusion mask types: box (blur and padding), occlusion (XSeg 1/2/3, 256 px, GPL-3.0), area (from 68 landmarks), and region (BiSeNet ResNet18/34, 512 px, MIT). They are combined with `numpy.minimum.reduce`. — [core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py); [face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
  - VisoMaster-Fusion adds Occluder plus XSeg with Mouth/Lips Protection, parser regions with grow sliders, Restore Eyes/Mouth blends, and Mouth Fit & Align. — [manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)
- **Mouth mask.** Deep-Live-Cam's Mouth Mask "retain[s] your original mouth for accurate movement". 2.7-beta added lip, chin and eyes masks, and 2.7 fixed lip-mask symmetry and ghosting. — [README](https://github.com/hacksider/Deep-Live-Cam); [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
- **Color and blending.**
  - Deep-Live-Cam: Poisson (`cv2.seamlessClone`) with the warped paste mask, plus feathering and color matching. — [face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py); [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
  - Rope: LAB color matching. — [Rope](https://github.com/Hillobar/Rope)
  - VisoMaster-Fusion: AutoColor and ending color transfer; RCT for DFM. — [manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)
- **Temporal stability.**
  - VisoMaster-Fusion: "Enable KPS Smoothing" (EMA on keypoints, with an alpha slider) and EMA on auto-mouth confidence. — [manual](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/user_manual.md)
  - Deep-Live-Cam: a Poisson mask derived from the affine, "no independent jitter source, no EMA, no lag", plus real-time frame interpolation in 2.7. — [face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py); [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
  - AlphaFace note: the pitch alignment templates can "pop between frames" near profile, so only the 5 yaw templates are used. — [alphaface.md](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/alphaface.md)
- **Identity/attribute balance.**
  - FaceFusion `face_swapper_weight` mixes up to 35% of the target embedding into the source. — [core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
  - Rope: "Likeness / Fidelity" settings. — [Rope](https://github.com/Hillobar/Rope)
- **Performance.**
  - Deep-Live-Cam: async detection (the camera never stalls), batched multi-face inference, and ROI-only paste. — [releases](https://github.com/hacksider/Deep-Live-Cam/releases)
  - deep_faceswap: ROI-based paste-back gave "~3.7x end-to-end on 720p". — [deep_faceswap](https://github.com/LdDl/deep_faceswap)
  - VisoMaster-Fusion: TensorRT FP16 plus cached engines. — [alphaface.md](https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/alphaface.md)

### Inferences
- For Rust, the cheap high-value tricks, roughly in order:
  1. Run detection asynchronously or every N frames, with landmark EMA or one-euro smoothing.
  2. Do affine warp, masking and paste-back only on the face ROI, ideally on the GPU (wgpu/CUDA).
  3. Use a cached feathered alpha template.
  4. Use ORT with CUDA graphs and IOBinding, or TensorRT FP16 with a cached engine.
  5. Batch the pixel-boost sub-inferences into one call.
  6. Add a BiSeNet (MIT) region/mouth mask.
  7. Add LAB/RCT color transfer.
- Poisson blending and an XSeg occluder are costlier on the CPU, and XSeg is GPL-3.0.

### Gaps
- I found no quantitative latency data for each trick (for example, the ms cost of BiSeNet or XSeg per frame, or of Poisson blending at 720p).
- Whether any project uses optical-flow-based temporal smoothing of the swapped crop itself was not found.
