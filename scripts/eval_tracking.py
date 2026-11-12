"""M2.5: render annotated ByteTrack videos on KITTI tracking clips. Runs on Kaggle (eval_track).

    python -m scripts.eval_tracking --config configs/tracker_eval.yaml            # 3 sequences
    python -m scripts.eval_tracking --config configs/tracker_eval.yaml --smoke    # 30 frames

For each sequence: ``Detector`` (torch backend, best.pt from W&B) -> ``Tracker`` (ByteTrack) ->
``draw_tracks`` -> MP4 (H.264 when ffmpeg is available, so W&B plays it in the browser).
Logs the videos, per-sequence track statistics and detector/tracker latency to W&B (group
``track``). MOT metrics (MOTA, IDF1, ID switches) are M2.6, not computed here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from scripts.eval_detection import fetch_weights, latency_stats

MAX_SEARCH_DEPTH = 6
IMAGE_SUFFIXES = (".png", ".jpg")


# ---------- pure helpers (unit-tested) ----------
def frame_paths(seq_dir: Path, max_frames: int | None) -> list[Path]:
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return frames[:max_frames] if max_frames else frames


def track_stats(ids_per_frame: Sequence[Sequence[int]]) -> dict[str, float]:
    """Simple id statistics; many short tracks hint at fragmentation / id switches."""
    lengths = Counter(i for ids in ids_per_frame for i in ids)
    active = [len(ids) for ids in ids_per_frame]
    return {
        "frames": len(ids_per_frame),
        "unique_ids": len(lengths),
        "mean_track_len": float(np.mean(list(lengths.values()))) if lengths else 0.0,
        "short_tracks_lt3": sum(1 for n in lengths.values() if n < 3),
        "mean_active": float(np.mean(active)) if active else 0.0,
        "max_active": max(active, default=0),
    }


def find_image_root(cfg: dict[str, Any]) -> Path:
    """KITTI tracking ``image_02`` folder: path from data_config, else a shallow search."""
    from perception.detection.detector import resolve_path

    kitti = yaml.safe_load(resolve_path(cfg["data_config"]).read_text())
    cand = Path(kitti["input_root"]) / kitti["tracking"]["image_dir"]
    if cand.is_dir():
        return cand
    search = Path(cfg["search"])
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        hits = sorted(search.glob("/".join(["*"] * depth) + "/training/image_02"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no training/image_02 under {search}; attach kitti-tracking")


# ---------- Kaggle-side steps ----------
def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 (browser-playable); even frame size is required by yuv420p."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-movflags", "+faststart", str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def render_sequence(
    seq: str, frames: list[Path], detector: Any, tracker: Any, cfg: dict[str, Any]
) -> dict[str, Any]:
    from perception.tracking.viz import draw_tracks

    out = Path(cfg["out_dir"])
    raw_path, final = out / f"track_{seq}_mp4v.mp4", out / f"track_{seq}.mp4"
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg["fps"], (w, h))
    tracker.reset()
    det_ms, trk_ms, ids_per_frame = [], [], []
    preview_at = len(frames) // 2
    for i, fp in enumerate(frames):
        frame = first if i == 0 else cv2.imread(str(fp))
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        tracks = tracker.update(dets, frame)
        t2 = time.perf_counter()
        if i >= cfg["latency_warmup"]:
            det_ms.append((t1 - t0) * 1000.0)
            trk_ms.append((t2 - t1) * 1000.0)
        ids_per_frame.append([t.track_id for t in tracks])
        img = draw_tracks(frame, tracks, trail=cfg["trail"])
        hud = (f"KITTI {seq} frame {i}  tracks {len(tracks)}  "
               f"det {(t1 - t0) * 1000:.1f} ms  track {(t2 - t1) * 1000:.1f} ms")
        cv2.putText(img, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        writer.write(img)
        if i == preview_at:
            cv2.imwrite(str(out / f"track_{seq}_preview.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    writer.release()
    if to_h264(raw_path, final):
        raw_path.unlink()
    else:
        print("ffmpeg/libx264 unavailable: keeping the mp4v file (may not play in browsers)")
        raw_path.replace(final)
    det_l, trk_l = latency_stats(det_ms), latency_stats(trk_ms)
    total = np.asarray(det_ms) + np.asarray(trk_ms)
    return {
        "sequence": seq,
        "video": str(final),
        **track_stats(ids_per_frame),
        "det_ms": det_l.get("mean_ms"),
        "track_ms": trk_l.get("mean_ms"),
        "fps_det_track": float(1000.0 / total.mean()) if total.size else None,
    }


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/tracker_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(cfg["smoke"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    import wandb

    from perception.detection import Detector, load_config
    from perception.tracking import Tracker

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], group=w["group"], job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""), config=cfg,
    )
    weights = fetch_weights(run, w["weights_artifact"], Path(cfg["weights_dir"]))
    det_cfg = load_config(cfg["detector_config"], weights=str(weights), **cfg["detector_overrides"])
    trk_cfg = load_config(cfg["tracker_config"], frame_rate=cfg["fps"])
    detector, tracker = Detector(det_cfg), Tracker(trk_cfg)
    run.config.update({"detector": det_cfg, "tracker": trk_cfg}, allow_val_change=True)

    root = find_image_root(cfg)
    print("KITTI image root:", root, flush=True)
    rows = []
    for seq in cfg["sequences"]:
        seq_dir = root / seq
        if not seq_dir.is_dir():
            print(f"WARNING: sequence {seq} not found under {root}; skipped", flush=True)
            continue
        stats = render_sequence(seq, frame_paths(seq_dir, cfg["max_frames"]), detector, tracker,
                                cfg)
        print(json.dumps(stats), flush=True)
        rows.append(stats)
        run.log({f"video/{seq}": wandb.Video(stats["video"], format="mp4")})
    if not rows:
        raise SystemExit("no KITTI sequence rendered")

    run.log({"track_stats": wandb.Table(columns=list(rows[0]),
                                        data=[list(r.values()) for r in rows])})
    det = [r["det_ms"] for r in rows if r["det_ms"] is not None]
    trk = [r["track_ms"] for r in rows if r["track_ms"] is not None]
    summary = {"det_ms": float(np.mean(det)) if det else None,
               "track_ms": float(np.mean(trk)) if trk else None}
    if det and trk:
        summary["fps_det_track"] = 1000.0 / (summary["det_ms"] + summary["track_ms"])
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    result = {"sequences": rows, "summary": summary, "minutes": (time.time() - t_start) / 60}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("summary:", summary, f"W&B run {run.url}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
