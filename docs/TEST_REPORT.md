# TEST_REPORT.md – Test results (AIML-04, M7.1)

| | |
|---|---|
| Date | 2026-11-19 |
| Code under test | `main` at `6162fcb` (fresh clone from GitHub), plus the QA branch `qa-m7-final` (adds `tests/test_make_demo_video.py`) |
| CI | GitHub Actions workflow `CI` (`.github/workflows/ci.yml`): Ubuntu 22.04, Python 3.10, `pip install -r requirements-dev.txt`, `ruff check .`, `pytest -q` on every push |
| CI on `main` | Green. Last three runs all passed; run [44396994322](https://github.com/Outsideorbit29/aiml04-perception/actions/runs/44396994322) (commit `6162fcb`): ruff "All checks passed!", **477 passed** in 7.8 s |
| CI on `qa-m7-final` | Green. Run [44398702728](https://github.com/Outsideorbit29/aiml04-perception/actions/runs/44398702728) (commit `a242749`): ruff clean, **482 passed** in 10.7 s (477 + 5 new) |
| Local, fresh clone (Windows, Python 3.14, CPU) | `ruff check .`: all checks passed. `pytest -q`: **474 passed, 3 skipped** in 6–10 s on `6162fcb`; 479 passed, 3 skipped with the new tests. The 3 skips are the MOT metric tests (`motmetrics` not installed locally; CI installs it and runs them). |

The local run needed `supervision` (ByteTrack) for the 16 tracker tests. It was installed with
`pip install --user --no-deps "supervision>=0.22,<0.27"` plus `scipy` and `defusedxml`. The
`--no-deps` flag stopped pip from also installing `opencv-python`, which would have overwritten the
existing `opencv-python-headless`. Without supervision those 16 tests fail locally with
`ModuleNotFoundError`. That is a missing local package, not a code fault, and CI installs
supervision from `requirements-dev.txt`.

## What the tests cover

All tests run on CPU with small fixtures (`tests/fixtures/`: 5 BDD images + labels and a 3 s KITTI
clip, 512 KB) or synthetic data. No test needs a GPU, PyTorch, a dataset, W&B, Kaggle, CARLA or
ROS2. The whole suite takes under 10 s.

| Area | Test file(s) | Tests | What is checked |
|---|---|---|---|
| Data contracts | `test_types.py` | 6 | xyxy box bottom-centre, lane polylines, `LaneResult.contains()` (mask vs polylines), `PerceptionFrame` lead-object selection |
| Data prep (M1) | `test_bdd_yolo.py`, `test_prepare_bdd.py`, `test_check_data_yaml.py` | 36 | BDD100K to YOLO conversion, class mapping, stratified subset, end-to-end converter on 3 synthetic images, data YAML resolution |
| Detection (M2) | `test_detector.py`, `test_eval_scripts.py`, `test_train_yolo.py` | 44 | Letterbox pre-processing, NMS and box decoding back to image xyxy, class map, backend switch (fake backends), eval and training helpers |
| Tracking (M2) | `test_tracker.py`, `test_eval_mot.py` | 25 | ByteTrack ids persist across frames, short gaps keep the id, low-score detections never start a track, new ids appear on the 3rd matched frame (shipped config), KITTI MOT helpers and MOTA/IDF1 on synthetic sequences |
| Lanes (M3) | `test_lanes.py`, `test_eval_lanes.py`, `test_yolop_engine.py` | 91 | Classical HLS + Sobel pipeline, YOLOP post-processing (mocked ONNX session, probs vs logits), single-side fallback, `LaneResult.contains()`, lane eval GT handling, `.engine` path selection |
| Depth and distance (M4) | `test_depth.py`, `test_eval_depth.py`, `test_distance_calibration.py`, `test_m5_distance_gate.py` | 58 | Flat-ground geometric distance, inverse-depth scale alignment, geometric-first fusion, detector-box calibration, KITTI LiDAR projection helpers, AbsRel metrics |
| Fusion, Kalman, TTC (M4.5) | `test_fusion.py` | 61 | Per-track Kalman filter (dt in seconds), closing speed, TTC gate (track age ≥ 0.75 s, closing > 2σ), stationary targets never get a short TTC, re-acquire restart, metric ego-lane corridor |
| Offline pipeline (M4.6) | `test_pipeline.py` | 11 | Stage wiring into `PerceptionFrame`, depth every N frames, strict JSONL, overlay drawing, video and image-folder sources, the CLI on the fixture clip (model loading mocked), import stays light |
| Control (M5.4) | `test_control.py` | 49 | AEB / ACC / GAP / HOLD rules, lead persistence over frames, hysteresis, dwell, standstill hold, creep to the gap, brake rate limit |
| CARLA and ROS2 glue (M5) | `test_sim.py`, `test_gt_labels.py`, `test_sim_eval.py`, `test_ros_conversions.py`, `test_train_mix.py` | 61 | Scenario logic and config consistency without the `carla` package, CARLA 3D box projection and occlusion, sim eval helpers, ROS message conversions and `.msg` definitions without ROS, BDD + CARLA mix split |
| ONNX / TensorRT (M6) | `test_deploy.py`, `test_trt_runtime.py` | 35 | Export and benchmark helpers, exported backends for detector, depth and YOLOP on CPU fakes (`tests/fake_trt.py`), TRT buffer handling, no torch import in the runtime path |
| Demo video (M7.4) | `test_make_demo_video.py` | 5 | `configs/demo_video.yaml` consistency (benchmark rows match the header, every clip is a run logged in EXPERIMENTS.md), card rendering, a full build on two synthetic clips (frame count, GIF), and an error on a missing clip |
| **Total** | 27 files | **482** | |

## How the numbers in the report were checked

Model quality and speed are measured on Kaggle, not in unit tests. Each run is logged in W&B and
in `docs/EXPERIMENTS.md`:

- Detection mAP: W&B `6euheeu2`, `8tzih6my`, `i6dnlk0c`.
- MOT: `qvrkp7t1`, `i3c2zmwu`.
- Lanes: `bf4xp164`.
- Distance: `amgl803i`, `7cnlg0dt`.
- Offline pipeline: `e12sh0xi`.
- TensorRT benchmark: `218me22j`.

For M7.1 the README headline numbers were checked against `docs/EXPERIMENTS.md` and
`docs/benchmarks.md`. The closed-loop table now uses the final Session C runs.

## Known gaps

- **No live CARLA or ROS2 tests in CI.** CI has no GPU, no CARLA server and no ROS2. The bridge,
  perception, control and recorder nodes are tested only through their ROS-free logic
  (`perception.control`, `perception_ros.conversions`, `sim/` helpers). The closed loop itself was
  checked by hand in four scenarios, each run once with the final configuration (Session C,
  `docs/EXPERIMENTS.md`).
- **No GPU backends in CI.** The TensorRT and ONNX Runtime paths run against CPU fakes. Real engine
  parity (TRT FP16 vs PyTorch) was measured on Kaggle (W&B `218me22j`).
- **Real model loading is mocked.** `Pipeline.from_config` is mocked in the CLI test. The torch
  path was last run end to end on Kaggle in M4.6 (W&B `e12sh0xi`, 2026-11-13), before the fusion
  and control changes of 2026-11-18/19. The TensorRT path ran live in Session C.
- **Untested scripts** (they need CARLA or full datasets):
  - `sim/record_dataset.py` and `sim/smoke_test.py` (CARLA).
  - `scripts/data/eda_bdd.py` and `scripts/data/probe_inputs.py` (one-off Kaggle data jobs).
- **Tuning is in-sample.** The tracker settings and the BDD lane warp were tuned on the data they
  are scored on, as `docs/EXPERIMENTS.md` notes.
- **No p95 latency for the live loop.** Only the means of 100-frame windows are logged.
