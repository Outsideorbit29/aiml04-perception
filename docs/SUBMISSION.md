# SUBMISSION.md – Deliverables checklist (AIML-04)

Each portal requirement mapped to the artifact that satisfies it.

Status updated 2026-11-19 (M7.4–M7.6, branch `m7-slides-reflection`, not merged yet). `[~]` = partly done.

| Portal requirement | Artifact | Location | Done | Status / owner |
|---|---|---|---|---|
| Working demo | 3–4 min screen recording of the CARLA scenarios with overlay | Drive `AIML-04/demo/` + link in README | [~] | All four scenarios recorded in Session B (`media/*_20261119_*.mp4`, local) and passed. Shot list, overlay guide and closing table: `docs/DEMO_SCRIPT.md`. TODO: re-record in a final session (sim-ros-integrator, M5.8), then edit and upload (user). README has marked placeholders for the GIF and video link. |
| GitHub repo with clean commits, README, setup | Public repo, `README.md`, `docs/SETUP.md` | GitHub | [~] | README refreshed with M5.9 and Session B results. Demo GIF missing (M7.4). M7.1 fresh-clone check by qa-reviewer still open. No LICENSE file (**user decision**). |
| Architecture diagram | `docs/diagrams/architecture.mmd` + `pipeline.mmd` (Mermaid, rendered by GitHub in `docs/diagrams/README.md` and README) | repo + Drive | [~] | Mermaid done (detector node now the mix model). PNG export for slides/Drive is TODO (**user**: Mermaid Live Editor in a browser, save as `docs/diagrams/architecture.png` and `pipeline.png`). |
| Data / API documentation | `docs/API.md` (ROS2 topics/messages, Python interfaces, config files); dataset notes in README "Datasets and licences" and `docs/figures/eda_summary.md` | repo | [x] | Done. |
| Test results | `pytest` report + `docs/EXPERIMENTS.md` | repo + Drive `AIML-04/reports/` | [~] | CI green on `main` (GitHub Actions). Exported pytest report for Drive is TODO (qa-reviewer, M7.1: e.g. `pytest -q --junitxml` artifact from CI). |
| Performance / accuracy numbers | mAP, MOTA/IDF1, lane F1, depth AbsRel, `docs/benchmarks.md`, closed-loop table | repo + slides | [~] | In README, `docs/EXPERIMENTS.md`, `docs/benchmarks.md`, slides 5–12. TBD: live-loop p95 latency not logged (sim-ros-integrator); `docs/benchmarks.md` laptop section still quotes the 2026-11-18 "40–45 ms" and not the Session B 35.0–44.0 ms windows (edge-optimizer). |
| 10-min presentation | `AIML-04_presentation.pptx/pdf` | Drive `AIML-04/` | [~] | Slide markdown with speaker notes and visuals: `presentation/slides.md` (14 slides, ~10 min). TODO (**user**): pandoc to PPTX, insert visuals (diagram PNG, demo clip), rehearse, export PDF. |
| Individual reflection | `docs/REFLECTION.md` (1 page) | repo + Drive | [~] | Scaffold: `docs/REFLECTION_DRAFT.md` (marked DRAFT). TODO (**user**): rewrite in own words as `docs/REFLECTION.md`. |
| Screenshots | `media/screenshots/` exported | Drive `AIML-04/screenshots/` | [ ] | TODO (**user**): export frames from the final CARLA recordings (one per scenario, plus the overlay freeze frame from `docs/DEMO_SCRIPT.md` shot 2); KITTI pipeline snapshots exist in the M4.6 Kaggle job output (W&B `e12sh0xi`). |
| Datasets | Sample subset + download scripts (not the full dataset) | Drive `AIML-04/data_sample/` | [ ] | TODO M7.7: `tests/fixtures/` (5 BDD images, 3 s KITTI clip) + `scripts/data/` + Kaggle dataset links. Drive upload is the user's action. |

## Drive folder plan (M7.7, upload is the user's action)

```
Anish_Kumar_Gupta_Capstone_Projects/          (share: Anyone with the link can view)
├─ AIML-04/
│  ├─ README.txt                    repo URL, demo link, one-paragraph summary, folder index
│  ├─ AIML-04_presentation.pdf      exported from presentation/slides.md (+ .pptx)
│  ├─ AIML-04_reflection.pdf        from docs/REFLECTION.md (the rewritten version)
│  ├─ code/
│  │  └─ aiml04-perception_<commit>.zip   `git archive` of the submitted commit (no models, data, media)
│  ├─ demo/
│  │  ├─ AIML-04_demo.mp4           edited 3–4 min video (docs/DEMO_SCRIPT.md)
│  │  └─ raw/                       the four final scenario recordings (optional)
│  ├─ reports/
│  │  ├─ EXPERIMENTS.pdf            docs/EXPERIMENTS.md
│  │  ├─ benchmarks.pdf             docs/benchmarks.md
│  │  ├─ API.pdf                    docs/API.md
│  │  ├─ architecture.png, pipeline.png   docs/diagrams exports
│  │  ├─ lanes_comparison.md, depth_comparison.md, lanes_failures.jpg, eda_*.png   docs/figures
│  │  └─ pytest_report.xml          CI junit artifact (M7.1)
│  ├─ screenshots/                  one PNG per scenario + overlay freeze frame + KITTI snapshot
│  └─ data_sample/
│     ├─ fixtures/                  copy of tests/fixtures/ (5 BDD images + labels, 3 s KITTI clip)
│     ├─ scripts_data/              copy of scripts/data/
│     └─ DATASETS.txt               links: Kaggle awsaf49/bdd100k-dataset, solesensei/solesensei_bdd100k,
│                                   garymk/kitti-3d-object-detection-dataset, leducnhuan/kitti-tracking;
│                                   licences (BDD100K non-commercial, KITTI CC BY-NC-SA 3.0); the CARLA
│                                   frames dataset is private on Kaggle (say so)
└─ AIML-01/                         second project (separate work)
```
Do not upload model weights, full datasets or `kaggle.json`.

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
