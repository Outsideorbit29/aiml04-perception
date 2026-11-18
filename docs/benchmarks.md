# Benchmarks (M6): PyTorch vs ONNX Runtime vs TensorRT

Official before/after table (ARCHITECTURE.md 3.8). Every number below was measured; nothing is
estimated. Reproduce with `kaggle kernels push -p kaggle/export_bench` (runs `deploy/export.py`,
`deploy/build_local_engine.py` and `deploy/benchmark.py` with `configs/bench.yaml`).

## T4 (Kaggle), batch 1 — 2026-11-18

Run: W&B [218me22j](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/218me22j)
(benchmark, group `bench`), export + parity [vhjwlvxw](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/vhjwlvxw);
code commit c52e607 (branch `m6-onnx-trt`).

| Model (input) | Backend | Model p50 ms | Model p95 ms | Model FPS | Speed-up vs PyTorch (p50) | Module p50 ms | Module FPS | Peak VRAM MB | Accuracy | Δ vs PyTorch | Parity max abs diff (fixtures) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s (BDD) (640x640) | PyTorch FP32 | 9.93 | 10.13 | 100.7 | 1.00x | 16.20 | 60.5 | 289 | mAP50 0.5132 / 50-95 0.2952 | ref | ref |
| YOLO11s (BDD) (640x640) | ONNX Runtime CUDA FP32 | 10.79 | 10.98 | 92.8 | 0.92x | 15.16 | 65.7 | 383 | mAP50 0.5132 / 50-95 0.2952 | +0.0000 / +0.0000 | boxes 0.00124, scores 2.86e-06 |
| YOLO11s (BDD) (640x640) | TensorRT FP16 | 3.66 | 3.71 | 273.3 | 2.72x | 9.60 | 102.7 | 181 | mAP50 0.5131 / 50-95 0.2952 | -0.0001 / +0.0001 | boxes 10.1, scores 0.0227 |
| MiDaS-small (256x448) | PyTorch FP32 | 10.95 | 11.51 | 90.6 | 1.00x | 14.75 | 67.2 | 291 | AbsRel vs ref 0 | ref | ref |
| MiDaS-small (256x448) | ONNX Runtime CUDA FP32 | 9.04 | 9.11 | 111.0 | 1.21x | 11.82 | 84.4 | 383 | AbsRel vs ref 0.0000 | +0.0000 | inv_depth 0.00738 |
| MiDaS-small (256x448) | TensorRT FP16 | 2.58 | 2.63 | 388.4 | 4.25x | 12.06 | 83.6 | 173 | AbsRel vs ref 0.0096 | +0.0096 | inv_depth 20.3 |
| YOLOP (640x640) | PyTorch FP32 | 22.78 | 23.78 | 43.7 | 1.00x | 26.07 | 38.1 | 249 | lane IoU vs ref 1 | ref | ref |
| YOLOP (640x640) | ONNX Runtime CUDA FP32 | 20.22 | 20.78 | 49.3 | 1.13x | 23.62 | 42.4 | 639 | lane IoU vs ref 1.0000 | -0.0000 | det_out 0.00409, drive_area_seg 2.25e-05, lane_line_seg 2.87e-05 |
| YOLOP (640x640) | TensorRT FP16 | 6.70 | 6.82 | 149.3 | 3.40x | 9.99 | 99.9 | 189 | lane IoU vs ref 0.9928 | -0.0072 | det_out 5.58, drive_area_seg 0.0619, lane_line_seg 0.0544 |

**Environment:** Tesla T4 (Kaggle, 15 GB), driver 580.178.04 (CUDA 13.0 driver API), torch 2.9.1
(CUDA 12.8, cuDNN 9.10.2), onnxruntime-gpu 1.32.0, TensorRT 10.16.1.11 (`tensorrt-cu12`),
cuda-python 12.9.10, ultralytics 8.4.215, timm 0.6.13.

**Method.**
- 200 warm-up + 1000 timed iterations per row, batch 1, one process per model and backend.
- **Model** = the network alone on a fixed preprocessed tensor from a 1280x720 BDD val frame, host
  numpy in and host numpy out (H2D/D2H copies included, as in the live pipeline).
- **Module** = the whole live call on that frame with the same runtime pre/post-processing for
  every backend: `Detector.detect` (letterbox, decode, NMS at the runtime conf 0.1),
  `DepthEstimator.estimate` (resize, normalise, resize back to 1280x720),
  `YolopLaneDetector.lane_binary` (letterbox, lane margin, resize).
- **Peak VRAM** = NVML peak device memory during the worker minus the idle reading before it,
  so CUDA context and allocator pools are included.
- **PyTorch** is FP32 eager: the fused Ultralytics `DetectionModel` behind the shared 640x640
  letterbox, MiDaS-small from torch.hub, and YOLOP `End-to-end.pth` from torch.hub.
- **ONNX Runtime** uses the CUDA EP in FP32. **TensorRT** is an FP16 engine built on this T4 by
  `deploy/build_local_engine.py` (1 GiB workspace, default optimisation level).
- **Accuracy.**
  - YOLO: mAP on 500 evenly spaced BDD val images, with val settings (conf 0.001, IoU 0.7,
    class-aware NMS) and Ultralytics-style matching and 101-point AP. The subset gives 0.513
    mAP@0.5, against 0.553 on the full 10k val; only the delta matters here.
  - MiDaS: mean per-pixel AbsRel of the raw inverse depth against PyTorch on 100 val frames.
    This is not LiDAR AbsRel. After per-frame scale alignment, a 1 % model difference is far
    below the 0.17 distance AbsRel from M4.
  - YOLOP: pooled lane-mask IoU against PyTorch on 100 val frames.
- **Parity** = max abs diff of every raw output against PyTorch on the 20 fixture frames
  (5 BDD images + 15 KITTI clip frames). YOLO box rows are in pixels.

**Reading the table.**
- **TensorRT FP16 is the fastest backend for all three models.** At model level it is 2.7x
  (YOLO), 4.3x (MiDaS) and 3.4x (YOLOP) faster than PyTorch FP32, with p95 within 2 % of p50.
  It also uses the least memory: 173–189 MB per model against 249–291 MB for PyTorch and
  383–639 MB for ONNX Runtime.
- **Accuracy cost of FP16 is negligible:** YOLO mAP@0.5 −0.0001, MiDaS 1 % relative, YOLOP
  lane mask IoU 0.993. The large YOLO box max-diff (10 px) comes from low-score anchors that
  NMS discards, as the unchanged mAP shows.
- **ONNX Runtime FP32 matches PyTorch to 4 decimals** in mAP and lane IoU (max abs diff: boxes
  1.2e-3 px, MiDaS 7.4e-3, YOLOP lane mask differs by 1 pixel). It is only
  1.1–1.2x faster for MiDaS and YOLOP and 8 % slower for YOLO at model level. Its YOLO module
  time is still 1 ms lower than PyTorch's; the cause was not profiled.
- **Module time is now CPU-bound for the TensorRT rows.**
  - YOLO: letterbox, decode and NMS take ~6 ms of 9.6 ms.
  - MiDaS: pre/post-processing (cubic resize of the 1280x720 frame, normalisation, resize of
    the map back to 1280x720) takes ~9.5 ms of 12.1 ms, so its module time equals ONNX
    Runtime's.
  - Live frames are smaller (CARLA 800x450), so these costs shrink there. Further speed-ups
    should come from preprocessing (e.g. INTER_LINEAR / INTER_AREA, or GPU preprocessing), not
    the model.
- YOLOP's shipped ONNX file (hustvl/YOLOP `yolop-640-640.onnx`, opset 12, sha256-pinned) is kept
  as the deployed model. Its parity with the PyTorch checkpoint is 3e-5 on the lane output, so
  the M3 lane numbers stay valid.

**Exported files** (W&B artifacts, project `aiml04-perception`):

| Artifact | File | Opset | Input | Source |
|---|---|---|---|---|
| `yolo11s-bdd-onnx:v0` | `yolo11s-bdd.onnx` (36 MB) | 17 | 1x3x640x640 | `yolo11s-bdd-best:latest`, Ultralytics export, onnxslim |
| `midas-small-onnx:v0` | `midas_small.onnx` (63 MB) | 17 | 1x3x256x448 | torch.hub intel-isl/MiDaS `MiDaS_small`, `torch.onnx.export` |
| `yolop-onnx:v0` | `yolop-640-640.onnx` (34 MB) | 12 | 1x3x640x640 | hustvl/YOLOP release file (verified against `End-to-end.pth`) |

T4 engine builds: YOLO 283 s, MiDaS 140 s, YOLOP 217 s (FP16, 1 GiB workspace); engines are
19–33 MB and are not stored (hardware-specific).

## Laptop end-to-end (RTX 3060 Laptop 6 GB, CARLA loop) — M6.4 / M6.5

Measured 2026-11-18 with engines built locally by
`python3 deploy/build_local_engine.py --onnx models/*.onnx --fp16` (WSL2).

**How to read these rows.** The live numbers are *one-frame overlay readings* of perception
**stage time only** (`1000 / total_ms` for that frame), not averages; MiDaS runs every 2nd
frame, so a single reading may exclude it; and the closed loop itself is capped at CARLA's
20 Hz. The ONNX Runtime row ran on the **CPU** (its CUDA provider could not load in WSL), so it
is not an ORT-on-GPU baseline. The averaged number to quote is the offline TensorRT run on 100
recorded CARLA frames: **66.8 FPS stage time, 33.8 FPS wall** (incl. decode/draw/encode).
Next session: log `total_ms` per frame and report mean and p95.

| Date | Backend | Pipeline FPS | GPU / driver / CUDA / TensorRT | Notes |
|---|---|---|---|---|
| 2026-11-18 | ONNX Runtime 1.22 | 7.6 (131 ms/frame) | RTX 3060 Laptop / 621.24 / – / – | Live `stopped_car` run with CARLA rendering. ORT's CUDA provider could not load (no cuBLAS/cuDNN in WSL), so detector and MiDaS ran on the CPU: detect alone 130 ms. Overlay reading mid-run. |
| 2026-11-18 | TensorRT FP16 (local engines) | 71.6 (14 ms/frame) | RTX 3060 Laptop / 621.24 / CUDA 12 (pip) / 10.16.1.11 | Same scenario, all three models on engines built by `deploy/build_local_engine.py` (MiDaS 721 s while CARLA was running, YOLO/YOLOP ~3 min each without it). Overlay reading near the stop. Offline on 100 recorded CARLA frames without CARLA: 66.8 FPS stages, 33.8 FPS incl. decode/draw/encode. |
