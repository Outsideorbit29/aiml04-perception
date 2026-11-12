"""Draw lane overlays on a few images or a short clip (fixture-sized; not for datasets).

Example (laptop, CPU, fixtures only; outputs go to the gitignored runs/ folder):
    python -m perception.lanes.render --images tests/fixtures/images --out runs/lanes_fixtures
    python -m perception.lanes.render --video tests/fixtures/clip_kitti.mp4 --out runs/lanes_clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from perception.lanes.base import LaneDetector, build_lane_detector, load_lane_config
from perception.lanes.fitting import measured_rows
from perception.types import LaneResult, Polyline

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _draw_polyline(
    img: np.ndarray,
    line: Polyline,
    y_top: float,
    color: list,
    thickness: int,
    y_bottom: float | None = None,
) -> None:
    h, w = img.shape[:2]
    ys = measured_rows(y_top, h - 1.0 if y_bottom is None else y_bottom, h)
    if ys.size < 2:
        return
    xs = np.clip(np.asarray(line.x_at(ys), dtype=np.float64), -w, 2 * w)
    pts = np.round(np.stack([xs, ys], axis=1)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, tuple(int(c) for c in color), thickness)


def draw_lanes(
    frame_bgr: np.ndarray,
    result: LaneResult,
    y_top: float,
    viz: dict,
    y_bottom: float | None = None,
    synthesized: Polyline | None = None,
) -> np.ndarray:
    """Return a copy of the frame with the ego-lane mask and both boundaries drawn.

    Boundaries are drawn on the same rows as the ego mask (`y_top`..`y_bottom`), so the overlay
    never shows the polynomial extrapolated below the measured region. `synthesized` (the
    single-side fallback boundary, `detector.last_debug["synthesized_line"]`) is drawn thin in
    `viz["synth_color"]` so it is never mistaken for a measured line.
    """
    out = frame_bgr.copy()
    if result.ego_mask is not None:
        tint = np.zeros_like(out)
        tint[result.ego_mask] = viz["mask_color"]
        alpha = float(viz["mask_alpha"])
        out[result.ego_mask] = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0)[result.ego_mask]
    thickness = int(viz["line_thickness"])
    if result.left is not None:
        _draw_polyline(out, result.left, y_top, viz["left_color"], thickness, y_bottom)
    if result.right is not None:
        _draw_polyline(out, result.right, y_top, viz["right_color"], thickness, y_bottom)
    if synthesized is not None:
        synth_t = int(viz["synth_thickness"])
        _draw_polyline(out, synthesized, y_top, viz["synth_color"], synth_t, y_bottom)
    return out


def _render(detector: LaneDetector, frame: np.ndarray, viz: dict) -> np.ndarray:
    result = detector.detect(frame)
    y_top = float(detector.last_debug.get("y_top", 0.0))
    y_bottom = detector.last_debug.get("y_bottom")
    syn = detector.last_debug.get("synthesized_line")
    return draw_lanes(frame, result, y_top, viz, y_bottom, syn)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/lanes.yaml")
    ap.add_argument("--images", help="folder of images (each treated as independent)")
    ap.add_argument("--video", help="short clip, processed with temporal smoothing")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--max-frames", type=int, default=60, help="safety cap, fixtures only")
    args = ap.parse_args(argv)

    cfg = load_lane_config(args.config)
    detector = build_lane_detector(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        folder = Path(args.images)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for path in files[: args.max_frames]:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            result = detector.detect(frame, stateless=True)
            dbg = detector.last_debug_stateless
            vis = draw_lanes(
                frame,
                result,
                dbg["y_top"],
                cfg["viz"],
                dbg["y_bottom"],
                dbg.get("synthesized_line"),
            )
            cv2.imwrite(str(out_dir / f"lanes_{path.stem}.jpg"), vis)
        print(f"wrote {min(len(files), args.max_frames)} overlays to {out_dir}")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg["viz"]["fallback_fps"])
        writer = None
        n = 0
        detector.reset()
        while n < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            vis = _render(detector, frame, cfg["viz"])
            if writer is None:
                h, w = vis.shape[:2]
                out_path = out_dir / f"lanes_{Path(args.video).stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            writer.write(vis)
            n += 1
        cap.release()
        if writer is not None:
            writer.release()
        print(f"wrote {n} video frames to {out_dir}")


if __name__ == "__main__":
    main()
