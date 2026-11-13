"""Full perception pipeline (M4.6): frame -> detect -> track -> lanes -> depth -> fusion.

    cfg = load_pipeline_config("configs/pipeline.yaml")
    pipe = Pipeline.from_config(cfg, backend="torch", camera_name="kitti", fps=10.0)  # Kaggle
    frame = pipe.process(img_bgr, stamp=t, frame_id=i)    # -> perception.types.PerceptionFrame
    vis = draw_frame(img_bgr, frame, cfg["viz"], pipe.lane_debug(), pipe.last_tracks)

Offline CLI (Kaggle job kaggle/run_pipeline, or any machine with the models):
    python -m perception.pipeline --source <video | image folder> --out out.mp4 \
        [--json frames.jsonl] [--summary summary.json] [--config configs/pipeline.yaml] \
        [--backend torch] [--camera kitti] [--lanes classical] [--max-frames N] [--fps 10]

The pipeline only composes the modules (perception.detection, .tracking, .lanes, .depth,
.fusion); it holds no perception logic of its own. Importing this module never imports torch,
onnxruntime, ultralytics or supervision: stages are built lazily by ``Pipeline.from_config``,
and tests inject their own stage objects.

Timings (``PerceptionFrame.timings_ms``): ``detect_ms``, ``track_ms``, ``lanes_ms``,
``depth_ms`` (wall time of each call, so frames that reuse the lane result or skip MiDaS show
the small amortised cost), ``fusion_ms`` (added by DistanceFusion) and ``total_ms``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from perception.depth.camera import CameraModel, load_camera
from perception.depth.estimator import merge_config
from perception.tracking.viz import id_color
from perception.types import LaneResult, ObjectState, PerceptionFrame, Track

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("detect_ms", "track_ms", "lanes_ms", "depth_ms", "fusion_ms", "total_ms")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Read configs/pipeline*.yaml; a top-level ``base: <path>`` is loaded first and merged."""
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = merge_config(load_pipeline_config(base), cfg)
    return cfg


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}


def stage_configs(
    cfg: dict[str, Any],
    backend: str | None = None,
    camera_name: str | None = None,
    lanes_method: str | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Per-module config dicts after the pipeline overrides (pure; no model is loaded).

    Order: module config <- ``overrides`` <- ``camera_overrides[camera]`` <- CLI arguments
    (``backend`` for detector + depth, ``lanes_method``, tracker ``frame_rate`` = ``fps``).
    Returns ``{"detector", "tracker", "lanes", "depth", "camera_name"}``.
    """
    from perception.lanes.base import load_lane_config  # cheap: no backend import

    cam = camera_name or cfg["camera"]
    ov = cfg.get("overrides") or {}
    cam_ov = (cfg.get("camera_overrides") or {}).get(cam) or {}

    def layered(base: dict[str, Any], key: str) -> dict[str, Any]:
        return merge_config(merge_config(base, ov.get(key) or {}), cam_ov.get(key) or {})

    from perception.depth.estimator import load_depth_config

    out = {
        "detector": layered(_read_yaml(cfg["detector_config"]), "detector"),
        "tracker": layered(_read_yaml(cfg["tracker_config"]), "tracker"),
        "lanes": layered(load_lane_config(cfg["lanes_config"]), "lanes"),
        "depth": layered(load_depth_config(cfg["depth_config"]), "depth"),
        "camera_name": cam,
    }
    if backend:
        out["detector"]["backend"] = backend
        out["depth"]["estimator"] = {**out["depth"].get("estimator", {}), "backend": backend}
    if lanes_method:
        out["lanes"]["method"] = lanes_method
    if fps and cfg.get("tracker_frame_rate_from_source", True):
        out["tracker"]["frame_rate"] = max(1, int(round(fps)))
    return out


class Pipeline:
    """Composes the stage objects; ``process(frame) -> PerceptionFrame``.

    Stages (any object with the same method works, which is how the tests run without models):
    ``detector.detect(frame)``, ``tracker.update(dets, frame)``, ``lanes.detect(frame)``,
    ``depth.maybe_estimate(frame)`` (None = no depth this frame; ``depth`` itself may be None),
    ``fusion.update(tracks, lanes, stamp, frame_id, hw, inv_depth, timings_ms)``.
    """

    def __init__(
        self,
        detector: Any,
        tracker: Any,
        lanes: Any,
        depth: Any | None,
        fusion: Any,
        fps: float,
    ) -> None:
        if not fps > 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.detector = detector
        self.tracker = tracker
        self.lanes = lanes
        self.depth = depth
        self.fusion = fusion
        self.fps = float(fps)
        self.last_tracks: list[Track] = []
        self.last_stats: dict[str, Any] = {}
        self.stage_cfgs: dict[str, Any] = {}  # set by from_config
        self._frame_idx = 0

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        backend: str | None = None,
        camera_name: str | None = None,
        lanes_method: str | None = None,
        fps: float | None = None,
        camera: CameraModel | None = None,
        detector_weights: str | None = None,
        yolop_weights: str | None = None,
    ) -> Pipeline:
        """Build every stage from ``cfg`` (heavy imports happen here, not at module import).

        ``camera`` replaces the configured camera model (e.g. KITTI per-sequence P2).
        """
        from perception.depth.estimator import DepthEstimator
        from perception.detection.detector import Detector
        from perception.fusion.fusion import DistanceFusion
        from perception.lanes.base import build_lane_detector
        from perception.tracking.tracker import Tracker

        fps = float(fps or cfg["source"]["fallback_fps"])
        sc = stage_configs(cfg, backend, camera_name, lanes_method, fps)
        if detector_weights:
            sc["detector"]["weights"] = detector_weights
        if yolop_weights:
            sc["lanes"].setdefault("yolop", {})["weights"] = yolop_weights
        cam = camera or load_camera(cfg["camera_config"], sc["camera_name"])
        pipe = cls(
            detector=Detector(sc["detector"]),
            tracker=Tracker(sc["tracker"]),
            lanes=build_lane_detector(sc["lanes"]),
            depth=DepthEstimator(sc["depth"]) if cfg.get("depth_enabled", True) else None,
            fusion=DistanceFusion(sc["depth"], cam),
            fps=fps,
        )
        pipe.stage_cfgs = sc
        return pipe

    def describe(self) -> dict[str, Any]:
        """Which backends actually run (logged by the Kaggle job; catches CPU fallbacks)."""
        det_cfg = getattr(self.detector, "cfg", {}) or {}
        depth_backend = getattr(self.depth, "backend", None)
        return {
            "detector_backend": det_cfg.get("backend"),
            "detector_weights": str(det_cfg.get("weights")),
            "tracker_frame_rate": (getattr(self.tracker, "cfg", {}) or {}).get("frame_rate"),
            "lanes_method": getattr(self.lanes, "name", type(self.lanes).__name__),
            "lanes_providers": list(getattr(self.lanes, "providers", []) or []),
            "depth_model": getattr(self.depth, "name", None),
            "depth_backend": type(depth_backend).__name__ if depth_backend else None,
            "depth_device": str(getattr(depth_backend, "device", "")) or None,
            "depth_every_n": getattr(self.depth, "every_n", None),
            "camera": repr(getattr(self.fusion, "camera", None)),
            "fps": self.fps,
        }

    def reset(self) -> None:
        """Forget all temporal state (call between unrelated clips)."""
        for stage in (self.tracker, self.lanes, self.depth, self.fusion):
            if stage is not None and hasattr(stage, "reset"):
                stage.reset()
        self.last_tracks = []
        self.last_stats = {}
        self._frame_idx = 0

    def lane_debug(self) -> dict[str, Any]:
        return dict(getattr(self.lanes, "last_debug", {}) or {})

    def process(
        self, frame: np.ndarray, stamp: float | None = None, frame_id: int | None = None
    ) -> PerceptionFrame:
        """One BGR frame. ``stamp`` defaults to frame index / fps, ``frame_id`` to the index."""
        if frame_id is None:
            frame_id = self._frame_idx
        if stamp is None:
            stamp = self._frame_idx / self.fps
        self._frame_idx += 1
        hw = (int(frame.shape[0]), int(frame.shape[1]))
        t = {}
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        t1 = time.perf_counter()
        tracks = self.tracker.update(dets, frame)
        t2 = time.perf_counter()
        lanes = self.lanes.detect(frame) if self.lanes is not None else None
        t3 = time.perf_counter()
        inv = self.depth.maybe_estimate(frame) if self.depth is not None else None
        t4 = time.perf_counter()
        t["detect_ms"] = 1e3 * (t1 - t0)
        t["track_ms"] = 1e3 * (t2 - t1)
        t["lanes_ms"] = 1e3 * (t3 - t2)
        t["depth_ms"] = 1e3 * (t4 - t3)
        out = self.fusion.update(tracks, lanes, stamp, frame_id, hw, inv_depth=inv, timings_ms=t)
        out.timings_ms["total_ms"] = 1e3 * (time.perf_counter() - t0)
        fstats = dict(getattr(self.fusion, "last_stats", {}) or {})
        self.last_tracks = list(tracks)
        self.last_stats = {
            "detections": len(dets),
            "tracks": len(tracks),
            "objects": len(out.objects),
            "no_estimate": int(fstats.get("no_estimate", 0)),
            "anchors": int(fstats.get("anchors", 0)),
            "aligned": int(fstats.get("aligned", 0)),
            "depth_ran": inv is not None,
        }
        return out


# ---------------------------------------------------------------------------- serialisation
def _num(v: Any) -> float | None:
    """JSON-safe float: numpy scalars cast, NaN/inf -> None."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _coeffs(line: Any) -> list[float] | None:
    return None if line is None else [_num(c) for c in np.asarray(line.coeffs).ravel()]


def object_to_dict(o: ObjectState) -> dict[str, Any]:
    d = o.track.det
    return {
        "track_id": int(o.track.track_id),
        "cls": int(d.cls),
        "cls_name": str(d.cls_name),
        "score": _num(d.score),
        "bbox_xyxy": [_num(v) for v in np.asarray(d.bbox_xyxy).ravel()],
        "age": int(o.track.age),
        "distance_m": _num(o.distance_m),
        "distance_std": _num(o.distance_std),
        "closing_speed_mps": _num(o.closing_speed_mps),
        "ttc_s": _num(o.ttc_s),
        "in_ego_lane": bool(o.in_ego_lane),
    }


def frame_to_dict(pf: PerceptionFrame, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialisable summary of a PerceptionFrame (the ego mask is reduced to its presence)."""
    lead = pf.lead_object()
    lanes = pf.lanes
    out = {
        "frame_id": int(pf.frame_id),
        "stamp": _num(pf.stamp),
        "timings_ms": {k: _num(v) for k, v in pf.timings_ms.items()},
        "lanes": {
            "left": _coeffs(lanes.left),
            "right": _coeffs(lanes.right),
            "confidence": _num(lanes.confidence),
            "has_ego_mask": lanes.ego_mask is not None,
        },
        "objects": [object_to_dict(o) for o in pf.objects],
        "lead_track_id": int(lead.track.track_id) if lead is not None else None,
    }
    if stats is not None:
        out["stats"] = {k: (bool(v) if isinstance(v, bool) else int(v)) for k, v in stats.items()}
    return out


def dumps_frame(pf: PerceptionFrame, stats: dict[str, Any] | None = None) -> str:
    """One JSONL line (strict JSON: no NaN)."""
    return json.dumps(frame_to_dict(pf, stats), allow_nan=False)


# ---------------------------------------------------------------------------- drawing
def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple, viz: dict) -> None:
    fs = float(viz["font_scale"])
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y = max(y, th + base)
    x = int(min(max(x, 0), max(0, img.shape[1] - tw)))
    cv2.rectangle(img, (x, y - th - base), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - base), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)


def object_label(o: ObjectState, lead: bool = False, compact: bool = False) -> str:
    """``#id class d+-std m TTC t s``; ``compact`` (small, far boxes) is ``#id d m``."""
    t = o.track
    if compact:
        return f"#{t.track_id} {o.distance_m:.0f}m"
    txt = f"#{t.track_id} {t.det.cls_name} {o.distance_m:.1f}+-{o.distance_std:.1f}m"
    if o.ttc_s is not None:
        txt += f" TTC {o.ttc_s:.1f}s"
    return ("LEAD " + txt) if lead else txt


def draw_frame(
    frame: np.ndarray,
    pf: PerceptionFrame,
    viz: dict[str, Any],
    lane_debug: dict[str, Any] | None = None,
    tracks: list[Track] | None = None,
    lane_viz: dict[str, Any] | None = None,
    info: list[str] | None = None,
) -> np.ndarray:
    """Annotated copy of ``frame``: lane overlay, tracks without distance (thin), objects with
    id/class/distance +- std/TTC, in-ego-lane outline, lead object, and an info block."""
    from perception.lanes.render import draw_lanes

    out = frame.copy()
    lanes = pf.lanes
    if lane_viz is not None and (
        lanes.ego_mask is not None or lanes.left is not None or lanes.right is not None
    ):
        dbg = lane_debug or {}
        if dbg.get("y_top") is not None:
            out = draw_lanes(
                out, lanes, float(dbg["y_top"]), lane_viz, dbg.get("y_bottom"),
                dbg.get("synthesized_line"),
            )
        else:  # no fit rows known: mask only, never an extrapolated polyline
            out = draw_lanes(out, LaneResult(None, None, lanes.ego_mask, 0.0), 0.0, lane_viz)
    obj_ids = {o.track.track_id for o in pf.objects}
    for t in tracks or []:
        if t.track_id in obj_ids:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in t.det.bbox_xyxy)
        color = id_color(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(viz["box_thickness"]))
        _label(out, f"#{t.track_id} {t.det.cls_name}", x1, y1, color, viz)
    lead = pf.lead_object()
    for o in sorted(pf.objects, key=lambda o: -o.distance_m):  # near objects drawn on top
        x1, y1, x2, y2 = (int(round(v)) for v in o.track.det.bbox_xyxy)
        color = id_color(o.track.track_id)
        is_lead = lead is not None and o is lead
        if o.in_ego_lane:
            th = int(viz["lead_thickness"]) if is_lead else int(viz["ego_lane_thickness"])
            outline = tuple(viz["lead_color"]) if is_lead else tuple(viz["ego_lane_color"])
            pad = int(viz["object_thickness"]) + th // 2  # outside the id-coloured box
            cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), outline, th)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(viz["object_thickness"]))
        warn = o.ttc_s is not None and o.ttc_s < float(viz["ttc_warn_s"])
        bg = tuple(viz["warn_color"]) if warn else (tuple(viz["lead_color"]) if is_lead else color)
        compact = not is_lead and (y2 - y1) < int(viz["full_label_min_h_px"])
        _label(out, object_label(o, is_lead, compact), x1, y1, bg, viz)
    lines = list(info or [])
    tm = pf.timings_ms
    if tm.get("total_ms"):
        total = tm["total_ms"]
        lines.append(f"frame {pf.frame_id}  {1e3 / total:.1f} FPS  {total:.0f} ms")
        lines.append("  ".join(f"{k[:-3]} {tm[k]:.0f}" for k in STAGES[:-1] if k in tm) + " ms")
    lines.append(f"objects {len(pf.objects)}  lane conf {lanes.confidence:.2f}")
    if lead is not None:
        lines.append(object_label(lead, True))
    step = int(viz["info_line_px"])
    fs = float(viz["font_scale"])
    for i, line in enumerate(lines):
        y = step * (i + 1)
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (0, y - th - base), (tw + 4, y + base // 2), tuple(viz["text_bg"]), -1)
        cv2.putText(
            out, line, (2, y - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
            tuple(int(c) for c in viz["text_color"]), 1, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------- sources
@dataclass
class FrameSource:
    """Frames of a video file or an image folder (sorted by name)."""

    fps: float
    count: int | None  # frames that will be yielded (after max_frames), None if unknown
    frames: Iterator[np.ndarray]


def open_source(
    source: str | Path,
    image_fps: float,
    fallback_fps: float,
    image_suffixes: list[str] | tuple[str, ...],
    max_frames: int | None = None,
) -> FrameSource:
    src = Path(source)
    suffixes = {s.lower() for s in image_suffixes}
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in suffixes)
        if max_frames:
            files = files[:max_frames]
        if not files:
            raise FileNotFoundError(f"no images {sorted(suffixes)} in {src}")

        def gen_images() -> Iterator[np.ndarray]:
            for p in files:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"cannot read image {p}")
                yield img

        return FrameSource(float(image_fps), len(files), gen_images())
    if not src.is_file():
        raise FileNotFoundError(f"source {src} is neither a video file nor a folder")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else float(fallback_fps)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    count = (min(n, max_frames) if max_frames else n) if n > 0 else None

    def gen_video() -> Iterator[np.ndarray]:
        try:
            i = 0
            while not max_frames or i < max_frames:
                ok, img = cap.read()
                if not ok:
                    break
                i += 1
                yield img
        finally:
            cap.release()

    return FrameSource(fps, count, gen_video())


# ---------------------------------------------------------------------------- summary
def summarize(records: list[dict[str, Any]], warmup: int, wall_s: float | None = None) -> dict:
    """Aggregate JSONL records: per-stage mean ms (after ``warmup`` frames), FPS, object stats."""
    n = len(records)
    timed = records[warmup:] if n > warmup else records
    mean_ms = {}
    for k in STAGES:
        vals = [r["timings_ms"][k] for r in timed if r["timings_ms"].get(k) is not None]
        if vals:
            mean_ms[k] = float(np.mean(vals))
    totals = [r["timings_ms"]["total_ms"] for r in timed if r["timings_ms"].get("total_ms")]
    objs = [o for r in records for o in r["objects"]]
    stats = [r.get("stats", {}) for r in records]
    with_dist = sum(s.get("objects", 0) for s in stats)
    no_est = sum(s.get("no_estimate", 0) for s in stats)
    dists = [o["distance_m"] for o in objs if o["distance_m"] is not None]
    ttc_objs = [o for o in objs if o["ttc_s"] is not None]
    ids = {o["track_id"] for o in objs}
    lanes_both = sum(1 for r in records if r["lanes"]["left"] and r["lanes"]["right"])
    lanes_mask = sum(1 for r in records if r["lanes"]["has_ego_mask"])
    out = {
        "frames": n,
        "timed_frames": len(timed),
        "warmup_frames": min(warmup, n),
        "mean_ms": mean_ms,
        "p50_total_ms": float(np.percentile(totals, 50)) if totals else None,
        "p95_total_ms": float(np.percentile(totals, 95)) if totals else None,
        "pipeline_fps": 1e3 / mean_ms["total_ms"] if mean_ms.get("total_ms") else None,
        "wall_fps": n / wall_s if wall_s else None,
        "detections_per_frame": float(np.mean([s.get("detections", 0) for s in stats]))
        if n
        else 0.0,
        "tracks_per_frame": float(np.mean([s.get("tracks", 0) for s in stats])) if n else 0.0,
        "objects_per_frame": len(objs) / n if n else 0.0,
        "frac_with_distance": with_dist / (with_dist + no_est) if with_dist + no_est else None,
        "unique_object_ids": len(ids),
        "in_ego_lane_object_frames": sum(1 for o in objs if o["in_ego_lane"]),
        "lead_frames": sum(1 for r in records if r["lead_track_id"] is not None),
        "ttc_count": len(ttc_objs),
        "ttc_tracks": len({o["track_id"] for o in ttc_objs}),
        "min_ttc_s": min((o["ttc_s"] for o in ttc_objs), default=None),
        "distance_m": {
            "min": float(np.min(dists)),
            "median": float(np.median(dists)),
            "max": float(np.max(dists)),
        }
        if dists
        else None,
        "lanes_both_sides_frac": lanes_both / n if n else 0.0,
        "lanes_ego_mask_frac": lanes_mask / n if n else 0.0,
        "depth_frames": sum(1 for s in stats if s.get("depth_ran")),
        "aligned_frames": sum(1 for s in stats if s.get("aligned")),
    }
    return out


# ---------------------------------------------------------------------------- run
def snapshot_indices(count: int | None, k: int) -> set[int]:
    """``k`` evenly spaced frame indices strictly inside [0, count) (first frames have no
    tracks yet, so the ends are skipped)."""
    if not count or k <= 0:
        return set()
    return {int(round(x)) for x in np.linspace(0, count - 1, k + 2)[1:-1]}


def run(
    pipe: Pipeline,
    src: FrameSource,
    cfg: dict[str, Any],
    out_mp4: str | Path | None,
    jsonl: str | Path | None = None,
    snapshots_dir: str | Path | None = None,
    lane_viz: dict[str, Any] | None = None,
    log_every: int = 0,
) -> dict[str, Any]:
    """Process every frame of ``src``; write the annotated video, JSONL and snapshot JPGs.
    Returns the summary dict (see ``summarize``)."""
    viz = cfg["viz"]
    ocfg = cfg["output"]
    snaps = snapshot_indices(src.count, int(ocfg["snapshots"])) if snapshots_dir else set()
    if snapshots_dir:
        Path(snapshots_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    jf = None
    if jsonl:
        Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
        jf = open(jsonl, "w", encoding="utf-8")  # noqa: SIM115 - closed in finally
    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    try:
        for i, img in enumerate(src.frames):
            pf = pipe.process(img, stamp=i / src.fps, frame_id=i)
            rec = frame_to_dict(pf, pipe.last_stats)
            records.append(rec)
            if jf:
                jf.write(json.dumps(rec, allow_nan=False) + "\n")
            if out_mp4 is None and i not in snaps:
                continue
            vis = draw_frame(img, pf, viz, pipe.lane_debug(), pipe.last_tracks, lane_viz)
            if out_mp4 is not None:
                if writer is None:
                    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*str(ocfg["fourcc"]))
                    writer = cv2.VideoWriter(str(out_mp4), fourcc, src.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer for {out_mp4}")
                writer.write(vis)
            if i in snaps:
                cv2.imwrite(
                    str(Path(snapshots_dir) / f"frame_{i:05d}.jpg"),
                    vis,
                    [cv2.IMWRITE_JPEG_QUALITY, int(ocfg["jpg_quality"])],
                )
            if log_every and (i + 1) % log_every == 0:
                print(f"frame {i + 1}: {pf.timings_ms.get('total_ms', 0):.1f} ms", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if jf:
            jf.close()
    wall = time.perf_counter() - t_start
    return summarize(records, int(cfg["summary"]["warmup_frames"]), wall)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="video file or image folder")
    ap.add_argument("--out", required=True, help="annotated MP4")
    ap.add_argument("--json", help="JSONL, one PerceptionFrame per line")
    ap.add_argument("--summary", help="summary JSON (default: printed only)")
    ap.add_argument("--snapshots", help="folder for evenly spaced annotated JPGs")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], help="detector + depth")
    ap.add_argument("--camera", help="camera name in configs/camera.yaml (default from config)")
    ap.add_argument("--lanes", choices=["classical", "yolop"], help="lane method override")
    ap.add_argument("--fps", type=float, help="image-folder fps (default source.image_fps)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--detector-weights", help="override detector weights path")
    ap.add_argument("--yolop-weights", help="override YOLOP ONNX path")
    ap.add_argument("--no-depth", action="store_true", help="geometric distance only")
    args = ap.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    if args.no_depth:
        cfg["depth_enabled"] = False
    s = cfg["source"]
    src = open_source(
        args.source,
        args.fps or float(s["image_fps"]),
        float(s["fallback_fps"]),
        s["image_suffixes"],
        args.max_frames,
    )
    pipe = Pipeline.from_config(
        cfg,
        backend=args.backend,
        camera_name=args.camera,
        lanes_method=args.lanes,
        fps=src.fps,
        detector_weights=args.detector_weights,
        yolop_weights=args.yolop_weights,
    )
    print(json.dumps(pipe.describe(), default=str), flush=True)
    summary = run(
        pipe, src, cfg, args.out, args.json, args.snapshots, pipe.stage_cfgs["lanes"]["viz"],
        log_every=50,
    )
    text = json.dumps(summary, indent=2, allow_nan=False)
    if args.summary:
        Path(args.summary).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return summary


if __name__ == "__main__":
    main()
