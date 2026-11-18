# SUBMISSION.md – Deliverables checklist (AIML-04)

Each portal requirement mapped to the artifact that satisfies it.

Status updated 2026-11-19 (M7.2/M7.3, branch `m7-docs`, not merged yet). `[~]` = partly done.

| Portal requirement | Artifact | Location | Done | Status / owner |
|---|---|---|---|---|
| Working demo | 3–4 min screen recording of the CARLA scenarios with overlay | Drive `AIML-04/demo/` + link in README | [ ] | TODO M7.4 (docs-presenter, after sim-ros-integrator records M5.8). Only `stopped_car` is recorded so far (`media/stopped_car_20261118_231618.mp4`, local). README has marked placeholders for the GIF and video link. |
| GitHub repo with clean commits, README, setup | Public repo, `README.md`, `docs/SETUP.md` | GitHub | [~] | README written (overview, results table with W&B ids, architecture, cloud workflow, usage, layout, licences, limitations). Demo GIF missing (M7.4). M7.1 fresh-clone check by qa-reviewer still open. No LICENSE file (main session to decide). |
| Architecture diagram | `docs/diagrams/architecture.mmd` + `pipeline.mmd` (Mermaid, rendered by GitHub in `docs/diagrams/README.md` and README) | repo + Drive | [~] | Mermaid done. PNG export for slides/Drive is TODO (user: export from Mermaid Live Editor in a browser; mermaid-cli needs headless Chromium, too heavy for the laptop). |
| Data / API documentation | `docs/API.md` (ROS2 topics/messages, Python interfaces, config files); dataset notes in README "Datasets and licences" and `docs/figures/eda_summary.md` | repo | [x] | Done, written from the code. Control thresholds deliberately not copied (being tuned). |
| Test results | `pytest` report + `docs/EXPERIMENTS.md` | repo + Drive `AIML-04/reports/` | [~] | CI green on `main` (GitHub Actions). Exported pytest report for Drive is TODO (qa-reviewer, M7.1: e.g. `pytest -q --junitxml` artifact from CI). |
| Performance / accuracy numbers | mAP, MOTA/IDF1, lane F1, depth AbsRel, `docs/benchmarks.md` | repo + slides | [~] | In README results table, `docs/EXPERIMENTS.md`, `docs/benchmarks.md`. TBD: closed-loop results for `pedestrian_crossing`, `cut_in`, `night_rain` (sim-ros-integrator, M5.5/M5.8); live-loop p95 latency not logged (sim-ros-integrator / edge-optimizer); M5.9 fine-tune result (running). Slides TODO (M7.5). |
| 10-min presentation | `AIML-04_presentation.pptx/pdf` | Drive `AIML-04/` | [ ] | TODO M7.5 (docs-presenter): slide markdown in `presentation/`, then PPTX. |
| Individual reflection | `docs/REFLECTION.md` (1 page) | repo + Drive | [ ] | TODO M7.6: docs-presenter drafts from the decisions log; the intern rewrites it in their own words. |
| Screenshots | `media/screenshots/` exported | Drive `AIML-04/screenshots/` | [ ] | TODO: user exports frames from the CARLA recordings; KITTI pipeline snapshots exist in the M4.6 Kaggle job output (W&B `e12sh0xi`). |
| Datasets | Sample subset + download scripts (not the full dataset) | Drive `AIML-04/data_sample/` | [ ] | TODO M7.7: `tests/fixtures/` (5 BDD images, 3 s KITTI clip) + `scripts/data/` + Kaggle dataset links. Drive upload is the user's action. |

## Portal form
- GitHub Repository URL: the repo root
- Live Deployment URL: a demo video link (YouTube unlisted or Drive), or a Hugging Face Space running the offline pipeline on uploaded video **(S)**
- Drive Folder URL: `Anish_Kumar_Gupta_Capstone_Projects` (both projects)
- Notes: results summary and how to run

**Submit only once, after AIML-01 is also ready.**

## Presentation outline (10 min)
1. Problem and goal (1 min)
2. Architecture and data flow (2 min)
3. Detection + tracking results (1.5 min)
4. Lanes: classical vs learned (1 min)
5. Depth and distance method + error (1 min)
6. CARLA + ROS2 closed-loop demo clip (1.5 min)
7. TensorRT before/after (1 min)
8. Challenges, learnings, next steps (1 min)
