# Demo video script (M7.4)

Target length **3:30** (limit 3–4 min). Edited from the CARLA closed-loop recordings, with short
title cards and a closing benchmark table. Every number on a card comes from
`docs/EXPERIMENTS.md` or `docs/benchmarks.md`.

> **Status.** The final recordings are from Session C (2026-11-19, PLAN.md M5.8); all four
> passed. `scripts/make_demo_video.py --config configs/demo_video.yaml` assembles the video from
> them; its cards and captions are the source of truth for the on-screen text. If a clip is
> re-recorded, update the cards from the new `docs/EXPERIMENTS.md` row.

## Source clips

| Scenario | Session C file (local, `media/`, not in git) | Logged result (EXPERIMENTS.md, Session C) |
|---|---|---|
| `stopped_car` | `stopped_car_20261119_013806.mp4` | ClearNoon; AUTOPILOT→ACC→AUTOPILOT→ACC to standstill; no AEB; min gap 11.8 m |
| `pedestrian_crossing` | `pedestrian_crossing_20261119_014523.mp4` | ClearNoon; AEB at TTC 1.49 s from 8.1 m/s → HOLD; min gap 9.6 m (final 8.5 m after pulling away) |
| `cut_in` | `cut_in_20261119_014200.mp4` | ClearNoon; car cuts in at 18 m, 5 m/s, then brakes 3 m/s²; ACC→AUTOPILOT→GAP to standstill; no AEB; min gap 10.0 m |
| `night_rain` | `night_rain_20261119_020707.mp4` | HardRainNight, lights on; AUTOPILOT→ACC→AUTOPILOT→GAP→ACC to standstill; lead read ~20 % long (13.6 m shown vs 11.3 m true); min gap 11.3 m |

All four: no collision. The loop runs in synchronous mode at 4.9 sim steps/s, so the recordings
play at **sim speed** (20 fps of sim time), not at the wall-clock speed of the session. Say so
once on screen.

## What the overlay shows (explain once, in shot 2)

From `perception.pipeline` (debug image) and `recorder_node` (bottom bar):

| Element | Meaning |
|---|---|
| Box label `#12 car 23.4+-1.1m TTC 3.2s` | Track id, class, fused distance ± its standard deviation, time to collision (shown only once the track is 0.75 s old and clearly closing) |
| `LEAD` prefix and outline | The object the controller reacts to: in the ego lane for 3 consecutive frames under one track id |
| Lane overlay | YOLOP lane lines and the ego-lane area |
| Top-left info block | Frame number, **one-frame** stage time (`FPS`, `ms`), per-stage ms (detect, track, lanes, depth, fusion), object count, lane confidence |
| Bottom bar (colour by mode) | Controller mode (AUTOPILOT / ACC / GAP / AEB / HOLD), brake command, ego speed in km/h, lead id and distance |

Caption to add over the info block in shot 2: "Per-frame stage time. Averaged over 100 frames the
live perception time was 38–49 ms; the whole loop ran at 4.9 sim steps/s." Do not quote the
one-frame FPS number from the overlay anywhere else (docs/benchmarks.md explains why).

## Shot list

| # | Time | Duration | Source | On screen | Voice-over / caption |
|---|---|---|---|---|---|
| 1 | 0:00 | 0:15 | Title card | "AIML-04 Autonomous Driving Perception Stack. Camera-only perception, CARLA + ROS2 closed loop." Small architecture diagram (`docs/diagrams/architecture.png`, TODO export). | "One front camera. Detection, tracking, lanes and distance run as ROS2 nodes on the laptop GPU, and a rule-based controller brakes for obstacles in the ego lane." |
| 2 | 0:15 | 0:25 | `stopped_car`, first seconds of driving, then a freeze frame | Freeze with callouts on: one box label, the LEAD tag, lane overlay, info block, bottom bar (table above). | "Each box shows track id, class, distance and time to collision. LEAD marks the object the controller reacts to. The bar at the bottom is the controller mode." Caption on the info block (above). |
| 3 | 0:40 | 0:40 | `stopped_car` | Approach to the stopped Audi A2 (placed 60 m ahead). Let the bar change AUTOPILOT → ACC → GAP. End on the standstill. | Card: "Stopped car, ClearNoon. ACC and gap control slow the car down; no emergency braking needed. Stopped 11.8 m short, no collision." |
| 4 | 1:20 | 0:40 | `pedestrian_crossing` | Walker steps out from 4 m beside the road. Slow down (0.5x) for the 2 s around AEB; the bar turns AEB then HOLD. | Card: "Pedestrian crossing. Emergency braking at TTC 1.5 s from 8.1 m/s, then hold. Stopped 9.6 m short, no collision." |
| 5 | 2:00 | 0:35 | `cut_in` | Car cuts in at 18 m, then brakes. Show the LEAD tag moving to the cut-in car and the bar changing ACC → AUTOPILOT → GAP. | Card: "Cut-in at 18 m, then the car brakes at 3 m/s². ACC, then gap control. Stopped 10.0 m behind it, no collision." |
| 6 | 2:35 | 0:30 | `night_rain` | HardRainNight approach, ACC stop. | Card: "Night, heavy rain. Stopped by ACC, 11.3 m short, no collision. At night the lead read about 20 % long (13.6 m shown, 11.3 m true)." |
| 7 | 3:05 | 0:25 | Benchmark card (static) | Table below. | "Models are exported to ONNX and run as TensorRT FP16 engines: two to four times faster than PyTorch on a T4, with no measurable accuracy loss." |
| | **3:30** | | | | |

If the cut runs long, shorten shot 3 to 0:30 and shot 5 to 0:30 (target 3:10). If it runs short,
add a 10 s card after shot 6 with the detection result: "Detector fine-tuned on BDD100K + CARLA:
CARLA-val mAP@0.5 0.819 (BDD-only model 0.624)."

## Closing benchmark card (shot 7)

From `docs/benchmarks.md` (T4, batch 1, model p50 latency, W&B `218me22j`):

| Model | PyTorch FP32 | ONNX Runtime | TensorRT FP16 | Speed-up |
|---|---|---|---|---|
| YOLO11s 640 | 9.93 ms | 10.79 ms | **3.66 ms** | 2.72x |
| MiDaS-small 256x448 | 10.95 ms | 9.04 ms | **2.58 ms** | 4.25x |
| YOLOP 640 | 22.78 ms | 20.22 ms | **6.70 ms** | 3.40x |

Footer line on the card: "FP16 cost: YOLO mAP@0.5 −0.0001, YOLOP lane-mask IoU 0.9928. Live on
the laptop with CARLA: 38–49 ms mean perception time per frame."

## Editing notes

- Editing runs on the laptop with a light editor (e.g. Clipchamp on Windows or Shotcut); export
  1280x720 or the native 800x450, H.264, under 100 MB.
- Do not speed up or slow down a clip without a "2x" / "0.5x" label.
- Title cards: white text on dark grey, at most two lines.
- Export a 10–15 s GIF from shot 3 for the README (`media/demo.gif`, then link or commit a
  compressed copy only if it is small; `media/` is gitignored).
- Upload: YouTube unlisted or Drive `AIML-04/demo/`; put the link in README "Demo" and in the
  portal "Live Deployment URL" field. Both are the user's actions.
