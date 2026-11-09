# Test fixtures

The only data files allowed on the laptop and in git (keep the total under 5 MB). Generated on
Kaggle by `python -m scripts.data.make_fixtures` (job `kaggle/data_prep`), never by hand.

| File | What | Source | Licence |
|---|---|---|---|
| `bdd_labels_sample.json` | 4 hand-written BDD-style frames (2018 + det_20 + per-frame layouts) for converter tests | written for the tests | project code |
| `images/bdd_*.jpg`, `labels/bdd_*.txt` | 5 BDD100K val images downscaled to 320x180, with YOLO labels (9 classes, `configs/data_bdd.yaml` order) | BDD100K 100k val split via Kaggle `awsaf49/bdd100k-dataset` | BDD100K licence, research / non-commercial use (see below) |
| `clip_kitti.mp4` | 30 consecutive frames (3.0 s at 10 fps), 480x144, mp4v | KITTI tracking, training sequence 0000, `image_02`, via Kaggle `leducnhuan/kitti-tracking` | CC BY-NC-SA 3.0 |
| `fixtures_manifest.json` | source image names, conditions, box counts, sizes | generated | – |

BDD100K: Copyright (c) 2018 The Regents of the University of California. Used for
educational, research and not-for-profit purposes under the BDD100K licence
(https://doc.bdd100k.com/license.html). F. Yu et al., "BDD100K: A Diverse Driving Dataset for
Heterogeneous Multitask Learning", CVPR 2020.

KITTI: A. Geiger, P. Lenz, C. Stiller, R. Urtasun, "Vision meets Robotics: The KITTI Dataset",
IJRR 2013. Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0
(https://creativecommons.org/licenses/by-nc-sa/3.0/). The clip is a downscaled excerpt.
