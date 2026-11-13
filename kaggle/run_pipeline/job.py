"""M4.6 Kaggle job: perception.pipeline on one KITTI tracking sequence (runs on Kaggle only).

    PYTHONPATH=. python kaggle/run_pipeline/job.py --config configs/pipeline_kaggle.yaml [--smoke]

Started by kaggle/run_pipeline/runner.py after cloning the repo. Steps: fetch the detector
weights (W&B artifact) and the YOLOP ONNX (yolop.weights_url), build the pipeline with the torch
backend for detector + MiDaS, run it on the sequence, then write to ``out_dir``:
out.mp4 (H.264 when ffmpeg is available), frames.jsonl, summary.json, snapshots/*.jpg, and log
the summary, snapshots and video to W&B (group ``pipeline``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_COMMIT")


def parse_p2(text: str) -> np.ndarray:
    """3x4 P2 from a KITTI calib file. Tracking calib writes ``P2: ...`` but ``R_rect ...`` /
    ``Tr_velo_cam ...`` without a colon, so scripts.eval_depth.parse_calib does not fit here."""
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if parts and parts[0] == "P2":
            return np.array(parts[1:13], dtype=np.float64).reshape(3, 4)
    raise KeyError("calib has no P2 line")


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/"))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/pipeline_kaggle.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    job = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        job.update(job["smoke"])
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    import wandb

    from perception.depth import load_camera
    from perception.lanes.yolop import ensure_weights
    from perception.pipeline import (
        Pipeline,
        load_pipeline_config,
        open_source,
        run,
        stage_configs,
    )
    from scripts.eval_detection import fetch_weights
    from scripts.eval_tracking import find_image_root, to_h264

    cfg = load_pipeline_config(job["pipeline_config"])
    w = job["wandb"]
    run_ = wandb.init(
        project=w["project"],
        entity=w.get("entity"),
        group=w["group"],
        job_type="smoke" if args.smoke else "pipeline",
        name=w["name"] + f"-{job['sequence']}" + ("-smoke" if args.smoke else ""),
        config={"job": job, "pipeline": cfg, "git_sha": git_sha()},
    )
    weights = fetch_weights(run_, w["weights_artifact"], Path(job["weights_dir"]))

    sc = stage_configs(cfg, job["backend"], job["camera"], job["lanes_method"], job["fps"])
    if sc["lanes"]["method"] == "yolop":
        ensure_weights({**sc["lanes"]["yolop"], "weights": job["yolop_weights"]})

    image_root = find_image_root({"data_config": job["data_config"], "search": job["search"]})
    seq_dir = image_root / job["sequence"]
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"sequence {job['sequence']} not found under {image_root}")
    src_cfg = cfg["source"]
    src = open_source(
        seq_dir, job["fps"], job["fps"], src_cfg["image_suffixes"], job["max_frames"]
    )
    camera = load_camera(cfg["camera_config"], job["camera"])
    calib_file = image_root.parent / "calib" / f"{job['sequence']}.txt"
    if job.get("use_calib_p2") and calib_file.is_file():
        first = cv2.imread(str(sorted(seq_dir.iterdir())[0]))
        p2 = parse_p2(calib_file.read_text())
        camera = camera.with_intrinsics(p2, first.shape[:2])
        print(f"camera intrinsics from {calib_file}", flush=True)
    else:
        print(f"no calib used ({calib_file}); camera {job['camera']} from config", flush=True)
    print(f"camera {camera}", flush=True)

    pipe = Pipeline.from_config(
        cfg,
        backend=job["backend"],
        camera_name=job["camera"],
        lanes_method=job["lanes_method"],
        fps=src.fps,
        camera=camera,
        detector_weights=str(weights),
        yolop_weights=job["yolop_weights"],
    )
    info = pipe.describe()
    print("pipeline", json.dumps(info, default=str), flush=True)
    run_.config.update({"stages": pipe.stage_cfgs, "describe": info}, allow_val_change=True)

    raw_mp4 = out / "out_mp4v.mp4"
    final_mp4 = out / "out.mp4"
    summary = run(
        pipe, src, cfg, raw_mp4, out / "frames.jsonl", out / "snapshots",
        pipe.stage_cfgs["lanes"]["viz"], log_every=50,
    )
    if to_h264(raw_mp4, final_mp4):
        raw_mp4.unlink()
    else:
        print("ffmpeg/libx264 unavailable: keeping the mp4v file (may not play in browsers)")
        raw_mp4.replace(final_mp4)

    summary.update(
        {
            "sequence": job["sequence"],
            "mode": "smoke" if args.smoke else "full",
            "git_sha": git_sha(),
            "describe": info,
            "wandb_url": run_.url,
            "job_seconds": time.time() - t_start,
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)

    run_.summary.update({k: v for k, v in flat(summary).items() if not isinstance(v, list)})
    table = wandb.Table(columns=["stage", "mean_ms"])
    for k, v in summary["mean_ms"].items():
        table.add_data(k, v)
    log = {"timings": table}
    snaps = sorted((out / "snapshots").glob("*.jpg"))
    if snaps:
        log["snapshots"] = [wandb.Image(str(p), caption=p.stem) for p in snaps]
    if final_mp4.exists():
        log["video"] = wandb.Video(str(final_mp4), format="mp4")
    run_.log(log)
    wandb.finish()


if __name__ == "__main__":
    main()
