# Individual reflection — DRAFT (AIML-04)

> **DRAFT. Do not submit this text as it is.** It is a scaffold written from the repo history,
> `docs/EXPERIMENTS.md` (results and decisions log) and `docs/benchmarks.md`. The portal asks for
> an *individual* reflection on what **you** learned and what **you** owned, so rewrite it in your
> own words, keep only what is true for you, and replace every `[fill in: ...]`. Delete this box
> when done. Target: one page (about 500–700 words). Save the final version as
> `docs/REFLECTION.md`.
>
> Facts below are from the docs. Feelings, anecdotes and opinions are left to you on purpose.

## What I built

I built a camera-only perception stack for the Nexvia AIML-04 brief. For every frame it detects
road users with YOLO11s fine-tuned on BDD100K, tracks them with ByteTrack, finds the ego lane with
YOLOP (compared against a classical OpenCV method), and estimates distance from flat-ground
geometry with MiDaS-small for coverage, plus a Kalman filter for closing speed and time to
collision. The core is one Python package with no ROS or CARLA code, so the same code runs on
dataset videos on Kaggle and live in CARLA through four ROS2 nodes. A rule-based controller
slows down and brakes for a lead object in the ego lane. In the final test session the car
stopped without a collision in all four scripted scenarios (stopped car, pedestrian crossing,
cut-in, night with heavy rain). The models run as TensorRT FP16 engines, 2.72x to 4.25x faster
than PyTorch on a T4.

[fill in: one or two sentences in your own words on what you are most satisfied with, and why.]

## What I owned

The repository was built between 2026-11-09 and 2026-11-19 (142 commits on `main` on 2026-11-19, before the slides and this draft). I used Claude Code
with project subagents (listed in `AGENTS.md`) for much of the code and documentation; the commit
trailers show this.

[fill in: be specific and honest. Which parts did you design or decide yourself? Which did you
write, review, debug or run yourself? Examples to check against your memory: the cloud-first
split, the dataset and class decisions, running every Kaggle job and reading its smoke output,
the CARLA sessions on your laptop, judging whether the numbers were plausible, deciding when a
milestone was done.]

[fill in: what you did not own or did not fully understand, if anything.]

## Decisions and trade-offs

These are from the decisions log in `docs/EXPERIMENTS.md`. Keep the ones you actually weighed and
say in your own words why.

- **Everything heavy runs in the cloud.** Training, evaluations, export and benchmarks ran as
  Kaggle jobs; the laptop only ran CARLA and ROS2. Cost: slower iteration and Kaggle's limits
  (about 30 GPU-hours per week, slow reads from mounted inputs). [fill in: was it worth it?]
- **Keep all 9 classes with repeat-factor sampling** instead of merging rare classes, because
  rider versus person matters for driving. Later I chose **not** to retrain with a stronger
  sampling setting, because a retrain cost about 4.6 GPU-hours for an uncertain gain and the
  deadline was shared with a second project.
- **ByteTrack with 2 consecutive frames before a new id.** ID switches fell from 310 to 176, at
  the cost of a 2-frame delay before a new track appears.
- **YOLOP as the live lane model,** classical as the CPU fallback: F1 0.515 vs 0.231, even though
  the ego-lane IoU was a tie.
- **MiDaS-small, not DPT-Hybrid:** no more accurate after alignment, and 27 ms vs 424 ms on a T4.
- **Geometric-first distance in CARLA.** The flat-ground model was near exact there (AbsRel 0.040
  on ground-truth boxes) and MiDaS made it worse (0.087), the opposite of KITTI.
- **A BDD + CARLA fine-tune for the simulator** (CARLA-val mAP@0.5 0.624 → 0.819, BDD val
  0.553 → 0.539). The BDD-only model stays the official real-data model.
- **Calibration back to identity** for the new detector, because the box-bottom calibration
  fitted for the old detector made the new one worse on every hold-out split.
- **Hysteresis, dwell and a 10 m minimum gap in the controller,** so ACC and gap control stop the
  car and emergency braking stays a backup.

[fill in: one decision you would defend in an interview, and one you are less sure about.]

## What I learned

Prompts drawn from what happened in the project. Pick two or three and write what *you* took
from them.

- Sim-to-real: the BDD fine-tune gained +0.20 mAP@0.5 on BDD but at most +0.045 on CARLA frames.
  [fill in]
- A fix fitted to one model does not carry over to the next: the calibration was
  detector-specific. [fill in]
- Run a small smoke job and look at its outputs before the full run; implausible numbers had
  meant evaluation bugs. [fill in]
- Averaging distances in metres biased fused distances short at range (−0.078 at 30–60 m on
  KITTI); fusing in inverse depth fixed it. [fill in]
- Deployment details matter: ONNX Runtime's CUDA provider would not load in WSL2, and with
  TensorRT the remaining time was CPU preprocessing, not the model. [fill in]
- Honest reporting: in-sample tuning, one run per scenario, in-domain validation. [fill in]
- [fill in: something about planning, time management with two projects, or working with AI
  coding agents.]

## What I would do differently

[fill in in your own words. Possible points from the record, keep only those you agree with:]

- Record CARLA frames earlier: the sim-to-real gap only showed up in M5, late in the schedule.
- Repeat each closed-loop scenario several times and test a second town, instead of one run each
  in Town01.
- Log p95 latency in the live loop from the start, not only the mean.
- Fit and check calibrations on route-level hold-outs from the start (the even/odd frame split
  shares route segments and looks better than it is).
- [fill in]

## Numbers I may quote (all from the docs)

| Area | Number |
|---|---|
| Detection (BDD val) | mAP@0.5 0.553; zero-shot 0.361 on 7 classes |
| Detection (CARLA-val, mix) | 0.819 vs 0.624 |
| Tracking (KITTI) | MOTA 52.5, IDF1 67.4, 176 ID switches |
| Lanes (BDD val 500) | F1 0.515 YOLOP vs 0.231 classical |
| Distance | AbsRel 0.170 KITTI (fused); 0.081 CARLA-val (mix, uncalibrated) |
| TensorRT (T4) | 3.66 / 2.58 / 6.70 ms; 2.72x / 4.25x / 3.40x |
| Closed loop | 4 of 4 scenarios stopped, no collision; gaps 10.1 / 9.6 / 10.2 / 15.5 m |
