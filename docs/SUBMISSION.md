# SUBMISSION.md – Deliverables checklist (AIML-04)

Each portal requirement mapped to the artifact that satisfies it.

Status updated 2026-11-27 (portal checked 2026-11-26). `[~]` = partly done.

## Portal facts (checked 2026-11-26)
- **Deadline: 2026-12-15** for both projects (was 2026-11-30). Plan: submit by 2026-12-14.
- The form takes **one public GitHub repo URL** holding both projects "as subfolders or branches
  (e.g. AIML-04 and AIML-01) with complete source code, tests, and screenshots".
  - AIML-04 Autonomous Driving Perception Stack: branch `main` of
    `https://github.com/Outsideorbit29/aiml04-perception`.
  - AIML-01 Real-Time Financial Fraud Detection: branch `AIML-01` of the same repo.
  - The README on `main` links to the `AIML-01` branch at the top so reviewers find both.
- The Google Drive folder URL and the Live Deployment URL are **optional** now.
- **Single submission only.** It is locked once submitted.

## Deliverables

| Portal requirement | Artifact | Location | Done | Status / owner |
|---|---|---|---|---|
| Working demo (deployed or screen-recorded) | Demo video built from the four final Session C recordings, plus the README GIF | `media/aiml04_demo.mp4` (1 min 56 s, local, gitignored); `docs/figures/demo.gif` in the README | [~] | Video built (M7.4, `python scripts/make_demo_video.py --config configs/demo_video.yaml`, shots in `docs/DEMO_SCRIPT.md`). It is shorter than the 3–4 min in the original plan. TODO (**user**): upload it (YouTube unlisted or Drive) and replace the TODO link in README "Demo". |
| GitHub repo with clean commits, README, setup | Public repo, `README.md`, `docs/SETUP.md`, `LICENSE` (AGPL-3.0) | GitHub `main` | [x] | README has the two-project note, results, screenshots and a License section. Fresh clone checked: ruff clean, pytest green (`docs/TEST_REPORT.md`); Kaggle fresh-clone pipeline smoke W&B `h31f403p` on `81a650e` (M7.1). |
| Screenshots (in the repo) | One frame per scenario with the overlay | `docs/screenshots/` (`stopped_car.jpg`, `cut_in.jpg`, `pedestrian_crossing.jpg`, `night_rain.jpg`), shown in README "Screenshots" | [x] | Extracted from the final Session C recordings (frame numbers in the README captions). |
| Architecture diagram | `docs/diagrams/architecture.mmd` + `pipeline.mmd` (Mermaid, rendered by GitHub in `docs/diagrams/README.md` and the README) | repo | [~] | Mermaid renders on GitHub, which meets the requirement. PNG export for the slides is still open (**user**, M7.3: Mermaid Live Editor, save as `docs/diagrams/architecture.png` and `pipeline.png`). |
| Data / API documentation | `docs/API.md` (ROS2 topics/messages, Python interfaces, config files); dataset notes in README "Datasets and licences" and `docs/figures/eda_summary.md` | repo | [x] | Done. |
| Test results | `docs/TEST_REPORT.md` (pytest/ruff counts, coverage by module, known gaps) + `docs/EXPERIMENTS.md` | repo | [x] | CI green on `main` (GitHub Actions). |
| Performance / accuracy numbers | mAP, MOTA/IDF1, lane F1, depth AbsRel, `docs/benchmarks.md`, closed-loop table | README "Results", `docs/EXPERIMENTS.md`, `docs/benchmarks.md`, slides | [~] | Done except **TBD**: per-frame p95 latency in the live loop is not logged (owner `sim-ros-integrator`). Minor (owner `edge-optimizer`): `docs/benchmarks.md` still has an older 2026-11-18 "40–45 ms" paragraph above the full all-sessions table (35.0–49.2 ms, median 40.6 ms). The two do not contradict each other, but the older paragraph could be removed. |
| 10-min presentation | Deck https://claude.ai/artifact/Gi42bagsTDZuMyVkVgz8UW (source `presentation/deck/`), slide markdown with notes `presentation/slides.md` | repo (+ PDF export) | [~] | TODO (**user**): fill the deck placeholders (demo video link, diagram PNG), rehearse to 10 min, export PDF. Slide 1 can use `docs/screenshots/stopped_car.jpg`. |
| Individual reflection | `docs/REFLECTION.md` (1 page) | repo | [~] | Scaffold: `docs/REFLECTION_DRAFT.md` (marked DRAFT). TODO (**user**, M7.6): rewrite in your own words as `docs/REFLECTION.md` and delete the draft box. |
| Datasets | Download/conversion scripts, small fixtures, Kaggle dataset links | `scripts/data/`, `kaggle/data_prep/`, `tests/fixtures/`, README "Datasets and licences" | [x] | Full datasets are linked, not committed. A Drive `data_sample/` is optional now (see below). |

## Remaining for the user (in order)
1. Rewrite `docs/REFLECTION_DRAFT.md` as `docs/REFLECTION.md` in your own words (M7.6).
2. Upload `media/aiml04_demo.mp4` (YouTube unlisted or Drive), put the link in README "Demo" and
   in the deck.
3. Export `docs/diagrams/architecture.mmd` and `pipeline.mmd` to PNG (M7.3) and fill the deck
   placeholders; rehearse and export the deck PDF.
4. Check that branch `AIML-01` is pushed, complete and has its own README, tests and screenshots.
5. Submit once on the portal by 2026-12-14 (deadline 2026-12-15).

## Optional: Drive folder (M7.7)
The Drive folder URL is optional on the portal now. If you still make one (for the video and the
deck PDF), use this layout. Upload is the user's action.

```
Anish_Kumar_Gupta_Capstone_Projects/          (share: Anyone with the link can view)
├─ AIML-04/
│  ├─ README.txt                    repo URL (branch main), demo link, one-paragraph summary
│  ├─ AIML-04_presentation.pdf      exported deck (+ .pptx if made)
│  ├─ AIML-04_reflection.pdf        from docs/REFLECTION.md (the rewritten version)
│  ├─ demo/
│  │  ├─ AIML-04_demo.mp4           media/aiml04_demo.mp4
│  │  └─ raw/                       the four final Session C recordings (optional)
│  ├─ reports/                      PDFs of docs/EXPERIMENTS.md, benchmarks.md, API.md, TEST_REPORT.md;
│  │                                diagram PNGs
│  ├─ screenshots/                  copy of docs/screenshots/
│  └─ data_sample/                  copy of tests/fixtures/ + DATASETS.txt (Kaggle links and licences:
│                                   BDD100K non-commercial, KITTI CC BY-NC-SA 3.0; CARLA frames dataset private)
└─ AIML-01/                         second project
```
Do not upload model weights, full datasets or `kaggle.json`. The local `.codex/` folder is
gitignored and must not go into any zip.

## Portal form
- GitHub Repository URL: `https://github.com/Outsideorbit29/aiml04-perception` (AIML-04 on `main`,
  AIML-01 on branch `AIML-01`; say this in the notes too)
- Live Deployment URL (optional): the demo video link, if uploaded. No Hugging Face Space was built
  (M7.8 stretch).
- Drive Folder URL (optional): `Anish_Kumar_Gupta_Capstone_Projects`, if made
- Notes: which branch holds which project, results summary, how to run

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
