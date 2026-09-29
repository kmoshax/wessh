# Auxiliary Face Models Around a Real-Time Face-Swap Pipeline (Rust + ONNX Runtime), state as of Sep 2026

Scope: detection, landmarks, recognition embeddings, parsing/occlusion masks, enhancers, temporal stability, mouth/expression preservation. The best single reference implementation is FaceFusion 3.x. Its source now attaches vendor, license and year metadata to every model. It downloads models from the `facefusion-assets` GitHub releases (`models-3.0.0`, `models-3.1.0`, `models-3.2.0`, `models-3.4.0`, `models-3.9.0` tags) via `resolve_download_url(tag, file)`. The exact host URL is built at runtime, so I did not verify the final URL string. The pattern is likely `https://github.com/facefusion/facefusion-assets/releases/download/<tag>/<file>`.

## 1. Face detection: which detector (speed, accuracy, license, ONNX)?

### Takeaway
SCRFD is still the best speed/accuracy detector with 5-point keypoints. Choose 2.5G_KPS for real time and 10G_KPS for quality. Its weights are InsightFace **non-commercial**. YuNet (MIT) is the only clean-license option that FaceFusion ships. It is tiny but clearly less accurate, especially on small or hard faces. The YOLO-face family is GPL-3.0 or AGPL-derived, so avoid it if the Rust app needs a permissive license.

### Cited Findings
- **FaceFusion 3.x detector list with metadata:** retinaface (`retinaface_10g.onnx`, InsightFace, Non-Commercial, 2020); scrfd (`scrfd_2.5g.onnx`, InsightFace, Non-Commercial, 2021); yolo_face (`yoloface_8n.onnx`, derronqi, **GPL-3.0**, 2022); yunet (`yunet_2023_mar.onnx`, OpenCV, **MIT**, 2023, added in models-3.4.0). All four return 5-point landmarks. Input normalization is [-1,1] for RetinaFace/SCRFD, [0,1] for YOLO-face and [0,255] for YuNet. FaceFusion also has `detect_faces_by_angle()`, which rotates the frame to catch rotated faces. — [FaceFusion face_detector.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_detector.py)
- **SCRFD official table (WIDER FACE Easy/Medium/Hard AP; FLOPs, latency and mAP all at VGA, 640×480):**
  - SCRFD_500M: 90.57 / 88.12 / 68.51, 0.57M params, **3.6 ms**
  - SCRFD_1G: 92.38 / 90.57 / 74.80, 4.1 ms
  - SCRFD_2.5G: 93.78 / 92.16 / 77.87, 0.67M params, **4.2 ms**
  - SCRFD_10G: 95.16 / 93.87 / 83.05, 3.86M params, **4.9 ms**
  - SCRFD_34G: 96.06 / 94.92 / 85.29, 9.80M params, 11.7 ms
  - KPS (5-keypoint) variants: 500M_KPS 90.97/88.44/69.49 (3.6 ms); 2.5G_KPS 93.80/92.02/77.13 (4.3 ms); 10G_KPS 95.40/94.01/82.80 (5.0 ms)

  The GPU used for these timings is not stated in the fetched README summary. These are 2021-era numbers. — [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- **InsightFace license:** "ALL models are available for non-commercial research purposes only" (model zoo header). This covers SCRFD, RetinaFace, ArcFace and the 2d106/3d68 landmark models. — [InsightFace model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- **YuNet (OpenCV Zoo):** WIDER Face val AP 0.834 / 0.824 / 0.708 (easy/medium/hard). It is designed for faces of about 10×10 to 300×300 px. The whole directory is **MIT**. Files:
  - `face_detection_yunet_2023mar.onnx` (fixed input)
  - `face_detection_yunet_2026may.onnx` (new in 2026, **dynamic input shape**, for OpenCV 5.x)
  - `_int8.onnx`
  - `_int8bq.onnx` (block-quantized, block 64)

  Paper: Wu et al., "YuNet: A tiny millisecond-level face detector", Machine Intelligence Research 2023. — [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **akanametov/yolo-face:** ships YOLOv8/v10/v11/v12 face models (n/s/m/l) trained on WIDERFace, including ONNX YOLOv12-n/s/m. It is built on Ultralytics. License: **GPL-3.0, or a paid Enterprise License** for commercial closed-source use. The README shows no keypoint (landmark) output, so a separate landmarker would be needed for alignment. — [akanametov/yolo-face](https://github.com/akanametov/yolo-face)
- **Other YOLO face variants:** YOLOv11n-face (225 epochs on WIDERFACE) is on HF at [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection). YOLO8Face/YOLO11Face with pose/keypoint heads (Ultralytics v8.3.75) are at [zjykzj/YOLO11Face](https://github.com/zjykzj/YOLO11Face).
- **MediaPipe Face Landmarker bundle:** includes a BlazeFace short-range detector at **192×192** input, float16. It is a short-range, selfie-distance detector. — [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

### Inferences
- For a webcam app (usually one large face), SCRFD_2.5G_KPS or even 500M_KPS at 640 input costs only a few ms, and the hard-set accuracy gap matters little. Detection is not the bottleneck; the swapper and enhancer are.
- Detection does not need to run every frame if a tracker carries the face between detections (see Section 6). That makes the detector choice even less performance-critical.
- **License-clean stack:** YuNet (MIT) plus a permissive landmarker. YuNet's 2026may dynamic-shape model simplifies arbitrary-resolution input in ORT. If you use the fixed-shape 2023mar model, you must resize to its fixed input.
- RetinaFace-10G gives no real benefit over SCRFD-10G for this use case. SCRFD dominates it on the speed/accuracy curve (inference from the SCRFD table plus FaceFusion defaults).

### Gaps
- I found no authoritative 2025–2026 side-by-side benchmark (same GPU/CPU, 640 input) of SCRFD vs YuNet vs YOLO11/12-face vs BlazeFace. The SCRFD table does not name its hardware.
- The Ultralytics AGPL-3.0 license was not re-verified in this session. akanametov's repo states GPL-3.0 plus an Enterprise license, and FaceFusion labels derronqi's yolov8-face GPL-3.0.
- YuNet benchmark ms numbers were not in the fetched README section.

## 2. Landmarks: which gives the most stable alignment for swapping?

### Takeaway
Alignment for inswapper, hyperswap and similar swappers uses only the **5 keypoints** from the detector, warped to an arcface_128 or ffhq_512 template. Dense landmarks (68/106/478) matter for masks (area mask, mouth mask, 3D pose/fade) and for stability, not for the core warp. FaceFusion's 68-point landmarkers:
- 2dfan4 (MIT)
- peppa_wutz (Apache-2.0)
- hrffa (MIT, new in 2025)
- fan_68_5 (5→68 estimator)

These are the practical permissive choices. MediaPipe's 478-point mesh (Apache-2.0) is good for masks and pose, but its geometry differs from the 68-point convention.

### Cited Findings
- **FaceFusion landmarker models:**
  - `2dfan4.onnx` (256×256; vendor breadbread1984; **MIT**; 2018)
  - `hrffa.onnx` (256×256; **PINTO0309; MIT; 2025**; models-3.9.0)
  - `peppa_wutz.onnx` (256×256; **Apache-2.0**; 2023)
  - `fan_68_5.onnx` (FaceFusion, OpenRAIL-M, 2024). It estimates 68 points from the 5 detector points and is used as a cheap fallback.

  In "many" mode FaceFusion runs both 2dfan4 and peppa_wutz and uses 2dfan4 when `score_2dfan4 > score_peppa_wutz - 0.2`, otherwise peppa_wutz. That is a confidence-based ensemble. — [FaceFusion face_landmarker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_landmarker.py)
- **InsightFace buffalo packs** (buffalo_l/m/s) include `2d106` and `3d68` alignment models (non-commercial). buffalo_sc omits them. — [InsightFace model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- **MediaPipe Face Landmarker:**
  - Output: 478 3D landmarks, 52 blendshape coefficients and facial transformation matrices
  - Models: face mesh model at 256×256; blendshape model with input 1×146×2
  - Running modes: IMAGE / VIDEO / LIVE_STREAM, with `min_tracking_confidence` (default 0.5). In VIDEO/LIVE_STREAM modes it tracks landmarks and skips re-detection while tracking confidence holds.

  — [MediaPipe Face Landmarker docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- **Deep-Live-Cam fork (beekamai/DeepLiveCam):** "68 landmarks fitted in 3D give the real yaw / pitch / roll". The swap fades out past configured angles, and "the 3D silhouette keeps the far cheek in the mask on turns". — [beekamai/DeepLiveCam](https://github.com/beekamai/DeepLiveCam)

### Inferences
- **Most stable alignment in practice:** 5-point keypoints from SCRFD, temporally filtered (One Euro, or EMA on the affine matrix; see Section 6). Heatmap 68-point models (2dfan4/hrffa) are more precise per frame but add a model call of about 256² per face. Run them only when a mask or pose feature needs them. Otherwise use fan_68_5 as a cheap approximation.
- MediaPipe's full pipeline (detector + mesh + tracking) is Apache-2.0 and designed for real-time tracking. Running the TFLite models from Rust would require converting them to ONNX (community conversions exist, not verified here) or using a TFLite runtime.

### Gaps
- I found no quantitative jitter or NME comparison of 2dfan4 vs peppa_wutz vs hrffa vs 2d106det vs MediaPipe for face-swap alignment stability.
- The PINTO0309 hrffa architecture and accuracy were not retrieved.

## 3. Recognition / identity embeddings

### Takeaway
Nearly every swapper still needs **InsightFace ArcFace w600k_r50** (512-D embedding, 112×112, non-commercial) as the source-identity encoder. Some swappers need an additional converter model (crossface_simswap, crossface_ghost, crossface_hififace) that maps ArcFace embeddings into their own space. Lightweight recognizers (MobileFaceNet/MBF, EdgeFace) are useful only for face *matching/selection*, e.g. picking which person in frame to swap. They cannot replace the embedding the swapper was trained on.

### Cited Findings
- **FaceFusion recognizer:** `arcface_w600k_r50.onnx`; InsightFace; **Non-Commercial**; 112×112; template `arcface_112_v2`; output is L2-normalized. — [FaceFusion face_recognizer.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_recognizer.py)
- **Swapper-to-embedding requirements (FaceFusion face_swapper):**

  | Swapper | Vendor | License | Template | Embedding path |
  |---|---|---|---|---|
  | inswapper_128 / _fp16 | InsightFace | Non-Commercial | arcface_128 | direct ArcFace; uses the model's internal emap matrix |
  | hyperswap_1a/1b/1c_256 | FaceFusion | ResearchRAIL | arcface_128 | direct |
  | alphaface_256 | AlphaFace | Non-Commercial | arcface_128 | direct |
  | simswap_256 / simswap_unofficial_512 | neuralchen | Non-Commercial | arcface_112_v1 | needs `crossface_simswap` |
  | ghost_1/2/3_256 | ai-forever | **Apache-2.0** | arcface_112_v1 | needs `crossface_ghost` |
  | hififace_unofficial_256 | GuijiAI | — | mtcnn_512 | needs `crossface_hififace` |
  | blendswap_256 | mapooon | Non-Commercial | ffhq_512 | uses the source frame instead of an embedding |
  | uniface_256 | xc-csc101 | — | ffhq_512 | uses the source frame instead of an embedding |

  — [FaceFusion face_swapper/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)
- **InsightFace packs:**
  - buffalo_l: SCRFD-10GF + ResNet50@WebFace600K, 326 MB; MR-ALL 91.25, LFW 99.83, IJB-C 97.25
  - buffalo_s: SCRFD-500MF + MBF (MobileFaceNet)@WebFace600K; MR-ALL 71.87, LFW 99.70, IJB-C 95.02
  - buffalo_sc: 16 MB, detection + MBF only
  - antelopev2: SCRFD-10GF + ResNet100@Glint360K, 407 MB
  - All non-commercial.

  — [InsightFace model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- **EdgeFace (Idiap, TBIOM 2024):** winner of the IJCB 2023 efficient-FR compact track (<2M params).
  - XXS: 1.24M params, about 94.7 MFLOPs, LFW 99.57
  - XS: 1.77M params, LFW 99.73, IJB-B 92.67, IJB-C 94.8
  - S: 3.65M params, LFW 99.78, IJB-C 95.63
  - Base: 18.23M params, LFW 99.83

  License **CC BY-NC-SA 4.0** (non-commercial). — [HF Idiap/EdgeFace-XXS](https://huggingface.co/Idiap/EdgeFace-XXS). ONNX exports: [yakhyo/edgeface-onnx](https://github.com/yakhyo/edgeface-onnx). The iPhone 15 Pro latency figures (XS 5.82 ms, S 9.89 ms) come from a search-result summary, not the primary source: [emergentmind EdgeFace](https://www.emergentmind.com/topics/edgeface).
- **FairFace classifier** (age/gender/race, used by FaceFusion for face selection filters): 224×224, CC-BY-4.0. — [FaceFusion face_classifier.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_classifier.py)

### Inferences
- **Embedding caching:** the source embedding is constant for a live session. Compute it once, or average it over several source photos. Only the target-side recognition (face selection or reference matching) runs per frame, and it can run every N frames and be tied to a track ID.
- **No permissive identity encoder is swapper-compatible.** inswapper and hyperswap are defined on w600k_r50's embedding space. A fully commercial-clean pipeline is not currently possible with the popular swappers. Ghost (Apache-2.0) is the exception, but it needs crossface_ghost, and the license of that converter/ArcFace backbone was not verified.
- AdaFace and other newer recognizers are not used by any swapper found. They could only serve matching.

### Gaps
- The license and origin of FaceFusion's `crossface_*` converter models were not retrieved.
- An AdaFace ONNX/licensing check was not completed. MobileFaceNet standalone (non-InsightFace) weights and licenses were not checked.

## 4. Face parsing and occlusion masks

### Takeaway
FaceFusion combines four mask types: box, occlusion (XSeg), area (landmark hull) and region (BiSeNet parsing). For live use, the usual choice is box plus one XSeg occluder at 256² (cheap). BiSeNet-ResNet18 at 512² is heavier and is only needed for region selection, e.g. keeping the target mouth, lips or eyes.

### Cited Findings
- **FaceFusion occluders:** `xseg_1`, `xseg_2` (models-3.1.0) and `xseg_3` (models-3.2.0), all **256×256** input. Several occluders can be combined with a minimum reduction.
- **FaceFusion parsers:** `bisenet_resnet_18` (the file is literally named `bisinet_resnet_18.onnx`, models-3.1.0) and `bisenet_resnet_34` (models-3.0.0), both **512×512**. The region mask is the argmax over parsing classes, then selected regions are kept.
- **FaceFusion mask types:** box (padding and blur); occlusion; area (convex hull of landmarks); region (parser).
- **FaceFusion post-processing for all masks:** GaussianBlur (sigma 5), clip to [0.5, 1], then `(v-0.5)*2`, which rescales to [0,1]. This gives a soft feathered edge.

  — [FaceFusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- **Temporal EMA on occlusion mattes:** one real-time swap project applies exponential moving averages across frames to occlusion probability mattes to remove "boundary edge shimmer and flicker". — [vasudev1876961/Real-Time-AI-Face-Swap](https://github.com/vasudev1876961/Real-Time-AI-Face-Swap)

### Inferences
- XSeg at 256² is roughly an order of magnitude less compute than BiSeNet at 512² (4× fewer pixels, and the network is presumably similar or smaller). For real time, run the occluder on the aligned crop only, and optionally every other frame with EMA smoothing in between.
- A landmark-based mouth or area mask (convex hull from 68/478 points) costs almost nothing and avoids the parser entirely for mouth preservation.

### Gaps
- No measured ms/frame for xseg_1/2/3 or BiSeNet ONNX was found. The xseg licenses/vendors were not in the fetched summary.
- I found no evidence of SAM-lite (MobileSAM / EfficientSAM / EdgeSAM) being used in face-swap occlusion pipelines, and no benchmarks for it. Not researched further.

## 5. Face enhancers/restorers faster than GFPGAN 1.4

### Takeaway
**GPEN-BFR-256** is the practical real-time enhancer:
- 8.5 ms with plain ORT CUDA and 4.8 ms with TensorRT in a Deep-Live-Cam fork
- GPEN has the fewest MACs of all surveyed restorers (17.78G at 512²)

GPEN-512, CodeFormer and RestoreFormer++ are around 40–60 ms at 512² on an RTX 3090 in PyTorch, which is not real-time alongside a swapper. Diffusion restorers (DiffBIR, DifFace) take seconds per image and are unusable live.

### Cited Findings
- **FaceFusion enhancer list** (default gfpgan_1.4, blend 80, weight 0.5):

  | Model | Template | Input |
  |---|---|---|
  | codeformer | ffhq_512 | 512 |
  | gfpgan_1.2 / 1.3 / 1.4 | ffhq_512 | 512 |
  | gpen_bfr_256 | arcface_128 | 256 |
  | gpen_bfr_512 | ffhq_512 | 512 |
  | gpen_bfr_1024 | ffhq_512 | 1024 |
  | gpen_bfr_2048 | ffhq_512 | 2048 |
  | restoreformer_plus_plus | ffhq_512 | 512 |

  — [FaceFusion face_enhancer/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_enhancer/core.py)
- **Survey Table 4** (512×512, RTX 3090; time in seconds / params / MACs):

  | Method | Time (s) | Params | MACs |
  |---|---|---|---|
  | GPEN | **0.04** | 26.23M | **17.78G** |
  | CodeFormer | 0.04 | 73.57M | 292.35G |
  | PSFR-GAN | 0.04 | 45.69M | 102.80G |
  | GFP-GAN | 0.06 | 60.76M | 85.04G |
  | RestoreFormer | 0.06 | 72.37M | 340.80G |
  | HiFaceGAN | 0.16 | — | — |
  | VQFR | 0.17 | — | 1067G |
  | STUNet | 0.17 | — | — |
  | DAEFR | 0.13 | — | — |
  | DFDNet | 0.66 | — | — |
  | **DifFace (diffusion)** | **11.38** | — | — |
  | OSDFace (one-step diffusion) | 0.71 in the table; the text says 0.13 s (internal inconsistency) | 865.79M | — |

  — [Wang et al., "Deep Face Restoration: A Survey", arXiv 2211.02831 (updated version)](https://arxiv.org/abs/2211.02831)
- **Deep-Live-Cam fork measurements**, full live pipeline:
  - Without TensorRT: HyperSwap 9.5 ms, Inswapper 19 ms, GPEN-256 8.5 ms → about 25 fps
  - With TensorRT: 4.4 ms / 6.2 ms / 4.8 ms → about 34 fps
  - The author calls the pipeline "launch-bound, not compute-bound"
  - Enhancer options: GPEN-256/512/1024, GFPGAN 1.4, CodeFormer, RestoreFormer++

  — [beekamai/DeepLiveCam](https://github.com/beekamai/DeepLiveCam)
- **Deep-Live-Cam upstream** offers GFPGAN, GPEN-256 and GPEN-512 enhancers as toggles. — [hacksider/Deep-Live-Cam](https://github.com/hacksider/deep-live-cam); [DeepWiki features](https://deepwiki.com/hacksider/Deep-Live-Cam/1.1-features-and-capabilities)
- **DiffBIR:**
  - About 8.01 s/image on A100 (search-result summary; I believe it comes from a comparison table in [InstantRestore, arXiv 2412.06753](https://arxiv.org/html/2412.06753v1) but did not verify the exact table)
  - About 1.0–1.5 s/image on H100 for DiffBIR v2.1 with 10 steps, FP16 (search-summary claim, source uncertain)

  Either way, it is confirmed far too slow for live use.
- **2026 state of the art:** the NTIRE 2026 Real-World Face Restoration challenge is dominated by quality-oriented, often diffusion-based methods, not real-time ones. — [NTIRE 2026 RWFR report, arXiv 2604.10532](https://arxiv.org/abs/2604.10532) (surfaced by search, not read in depth)

### Inferences
- **Frame-interleaving tricks that follow from the evidence:**
  1. Use GPEN-256 (its arcface_128 template matches inswapper and hyperswap crops, so no re-warp is needed). Run it on the aligned swap crop before paste-back.
  2. Run the enhancer asynchronously. beekamai's "low-latency reprojection" moves "the last finished swap … onto the newest camera frame by the tracker, so a slow enhancer delays the expression, not the position."
  3. Blend the enhancer at less than 100% (FaceFusion's default blend is 80) to reduce identity drift and flicker.
- CodeFormer/RestoreFormer++ could reach roughly 10–20 ms with TensorRT FP16 on high-end GPUs. That is unverified, and their 290–340G MACs make them poor for a lightweight app.

### Gaps
- There are no ORT-CUDA/DirectML/CPU ms numbers for GPEN-512, CodeFormer or RestoreFormer++ ONNX at batch 1.
- DMDNet timing was not found.
- No genuinely new 2025–2026 *lightweight* face restorer with ONNX weights was found.
- Licenses: GPEN (Alibaba) and CodeFormer (S-Lab License, non-commercial) were not re-verified this session.

## 6. Temporal stability for live video

### Takeaway
The state of the art in open live-swap apps (2025–2026) combines:
1. Detection every N frames, with **optical-flow tracking** in between (forward-backward check + RANSAC)
2. **Adaptive smoothing** (One Euro or velocity-weighted EMA) of keypoints or directly of the 2×3 affine matrix
3. A deadband that reuses the cached affine when motion is under about 1 px
4. EMA on occlusion mattes
5. Luminance-pop clamping and colour transfer
6. A fade-out, not a hard cut, when tracking is lost

Frame-by-frame independent processing is the main cause of flicker.

### Cited Findings
- **beekamai/DeepLiveCam:**
  - Tracking: "optical-flow tracking between detections: forty support points over the whole head, forward-backward check, RANSAC", which keeps hands and microphones from dragging the alignment
  - Lost faces: the face "is carried for a second after the detector loses it (deep turn) and fades out instead of cutting"
  - Calibration: gives an "expression-proof crop (pursed or smiling lips no longer shrink the swap)"
  - Pose: the swap fades past chosen yaw/pitch limits using 3D-fitted 68-point landmarks
  - Latency: low-latency reprojection of the last finished swap onto the newest frame

  — [beekamai/DeepLiveCam](https://github.com/beekamai/DeepLiveCam)
- **vasudev1876961/Real-Time-AI-Face-Swap** (SCRFD-500m + ArcFace + inswapper-style):
  - "adaptive velocity-weighted exponential smoothing to the 2x3 affine transformation matrix and clamps frame-to-frame luminance pops"
  - Default `ema_alpha: 0.65`
  - Deadband: frames with under 1.25 px shift reuse cached forward and inverse affine matrices
  - EMA on occlusion mattes
  - Claimed CPU baseline: 60.9 fps at 640×480, 37.2 fps at 720p, 25.9 fps at 1080p. These are self-reported and the hardware is unspecified.

  — [Real-Time-AI-Face-Swap](https://github.com/vasudev1876961/Real-Time-AI-Face-Swap)
- **One Euro filter:** an adaptive low-pass filter that favours noise reduction when the signal is slow and lag reduction when it moves fast. Raw landmarks jitter by 1–2 px on a still face, which becomes visible once it drives a warp. — [FreeFaceMoCap One Euro notes](https://mohamedalirashad.github.io/FreeFaceMoCap/2021-12-25-filters-for-stability/); [MediaPipe Face Mesh overview](https://www.emergentmind.com/topics/mediapipe-face-mesh)
- **MediaPipe** VIDEO/LIVE_STREAM modes use tracking with `min_tracking_confidence` to avoid re-detection each frame. — [MediaPipe docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- **Deep-Live-Cam colour correction:** automatic colour transfer from target to swapped face. — [DeepWiki Deep-Live-Cam](https://deepwiki.com/hacksider/Deep-Live-Cam/1.1-features-and-capabilities)
- **Offline research on video-consistent swapping** (not real-time, but useful for ideas): CanonSwap (canonical-space modulation, [arXiv 2507.02691](https://arxiv.org/html/2507.02691v1)); "Preserving Source Video Realism" ([arXiv 2512.07951](https://arxiv.org/html/2512.07951v1)); SmoothNet as a plug-in temporal refiner for pose sequences ([arXiv 2112.13715](https://arxiv.org/pdf/2112.13715)).

### Inferences
- **Recommended Rust design:**
  1. Run SCRFD every 3–10 frames or on tracker failure
  2. Use KLT/LK optical flow on support points in between (e.g. `opencv` crate or a hand-rolled pyramid LK)
  3. Apply a One Euro filter per keypoint coordinate, or on the similarity-transform parameters (scale, rotation, translation), which is more stable than filtering points independently
  4. Use a deadband for sub-pixel motion
  5. Apply an EMA on the final paste mask
  6. Compute colour statistics (mean/std in LAB) with EMA instead of per frame
  7. Keep a cached source embedding
  8. Associate faces across detections with simple IoU matching (ByteTrack-lite is enough for 1–few faces)
- Smoothing the affine matrix rather than raw points avoids shape distortion. Filtering with speed-adaptive cutoff (One Euro) avoids lag on fast head turns.

### Gaps
- I found no peer-reviewed paper specifically on *real-time* temporal consistency for one-shot face swap. Evidence is from open-source repos and self-reported numbers.
- ByteTrack-lite and Kalman specifics for faces were not researched.

## 7. Mouth/lip preservation, expression restoration, age/lighting correction

### Takeaway
Two complementary approaches:
- **Mouth mask:** paste the real target mouth back through a feathered mask bounded by landmarks. It is cheap and fixes tongue, teeth and hand-to-mouth artifacts. It is standard in Deep-Live-Cam.
- **Expression restoration:** FaceFusion's LivePortrait-based restorer (MIT, 512²) re-injects the target expression. It is heavier: three LivePortrait networks per face.

The eye region can be treated the same way as the mouth, taking real blinks from the camera. Colour transfer handles lighting. No mainstream real-time age correction model was found.

### Cited Findings
- **Deep-Live-Cam mouth mask:** "shows your real mouth instead of the deepfaked mouth", for tongue and touching the lips. — [hacksider/Deep-Live-Cam](https://github.com/hacksider/deep-live-cam); [DeepWiki](https://deepwiki.com/hacksider/Deep-Live-Cam/1.1-features-and-capabilities)
- **beekamai fork mouth and eye handling:**
  - The mouth mask is "bounded by features (lips → nose base → chin): a moustache is never cut in half"
  - Eyes: "the swap models only squint; during a blink the eye region comes from the camera, optionally all the time (your own gaze)"

  — [beekamai/DeepLiveCam](https://github.com/beekamai/DeepLiveCam)
- **FaceFusion expression_restorer:**
  - Uses LivePortrait feature extractor, motion extractor and generator; KwaiVGI, **MIT**, 2024; 512×512; arcface_128 template
  - Blends the expression as `target_expression * factor + temp_expression * (1 - factor)`, with the factor mapped from 0–100 to 0–1.2
  - Selectable areas are upper-face and lower-face

  — [FaceFusion expression_restorer/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/expression_restorer/core.py)
- **FaceFusion region masks** from BiSeNet can keep target regions such as mouth, lips or eyes, as an alternative to a landmark mouth mask. — [face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- **FaceFusion's face_swapper core** contains no explicit mouth-mask or colour-transfer logic. Those are separate mask or processor options. — [face_swapper/core.py](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)

### Inferences
- For a lightweight live app, use a landmark-hull mouth mask (and optionally an eye mask during blinks), feathered, with EMA on the mask. That gives most of the expression fidelity gain at negligible cost. The LivePortrait expression restorer (3 models at 512²) is better suited to offline or high-end GPU modes.
- Lighting consistency: use LAB mean/std colour transfer with temporal EMA of the statistics, plus luminance-pop clamping as in the vasudev repo.

### Gaps
- No ms/frame timing for FaceFusion's expression restorer or LivePortrait ONNX was found.
- FaceFusion's age_modifier processor (age correction) was not fetched.
- I found no real-time relighting model used in open face-swap apps.
