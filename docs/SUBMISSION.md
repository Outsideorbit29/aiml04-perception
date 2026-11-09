# SUBMISSION.md – Deliverables checklist (AIML-04)

Each portal requirement mapped to the artifact that satisfies it.

| Portal requirement | Artifact | Location | Done |
|---|---|---|---|
| Working demo | 3–4 min screen recording of the CARLA scenarios with overlay | Drive `AIML-04/demo/` + link in README | [ ] |
| GitHub repo with clean commits, README, setup | Public repo, `README.md`, `docs/SETUP.md` | GitHub | [ ] |
| Architecture diagram | `docs/diagrams/architecture.png` (from ARCHITECTURE.md section 2) | repo + Drive | [ ] |
| Data / API documentation | `docs/API.md` (ROS2 topics/messages, Python interfaces, dataset notes) | repo | [ ] |
| Test results | `pytest` report + `docs/EXPERIMENTS.md` | repo + Drive `AIML-04/reports/` | [ ] |
| Performance / accuracy numbers | mAP, MOTA/IDF1, lane F1, depth AbsRel, `docs/benchmarks.md` | repo + slides | [ ] |
| 10-min presentation | `AIML-04_presentation.pptx/pdf` | Drive `AIML-04/` | [ ] |
| Individual reflection | `docs/REFLECTION.md` (1 page) | repo + Drive | [ ] |
| Screenshots | `media/screenshots/` exported | Drive `AIML-04/screenshots/` | [ ] |
| Datasets | Sample subset + download scripts (not the full dataset) | Drive `AIML-04/data_sample/` | [ ] |

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
