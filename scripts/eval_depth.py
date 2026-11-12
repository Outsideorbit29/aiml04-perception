"""M4.4: object distance vs KITTI LiDAR. Runs on Kaggle (kaggle/eval_depth).

    python -m scripts.eval_depth --config configs/depth_eval.yaml            # 1000 frames
    python -m scripts.eval_depth --config configs/depth_eval.yaml --smoke    # 24 frames

Data: KITTI 3D object training split (image_2, calib, label_2, velodyne; paths from
configs/data_kitti.yaml). ``sample.n_frames`` frames are drawn with a fixed seed.

Objects: KITTI GT 2D boxes (``objects`` filters: class map, min height, max occlusion and
truncation) are used as detections with score 1, so the numbers measure distance estimation
only, not detection. Every estimator sees the same boxes.

Ground truth per object:
* ``lidar`` (primary): velodyne points -> rectified camera frame (R0_rect * Tr_velo_to_cam,
  both padded to 4x4) -> image_2 (P2). Points with z <= ``gt.min_depth_m`` are dropped before
  projecting. Points inside the 2D box AND inside the 3D label box (enlarged by
  ``box3d_margin_m``, ground hits within ``ground_clearance_m`` of the box bottom dropped) are
  the object's surface; GT = ``gt.percentile`` of their camera z. Objects with fewer than
  ``gt.min_points`` such points have no LiDAR support: counted (by bin, class, occlusion) and
  not scored.
* ``label_z`` (secondary): label_2 ``location`` z, the bottom centre of the 3D box (camera
  frame, y down). It is about half the object length behind the visible surface, so all
  methods look biased short against it; reported for completeness and as a geometry check
  (``lidar_vs_label``: median label z - LiDAR z per class).
Both are camera-frame depth z, not Euclidean range.

Methods (all metric distances, MiDaS is never reported unaligned):
* ``geometric``: flat-ground distance from the box bottom row; camera height, pitch and sigmas
  from configs/camera.yaml ``kitti``, intrinsics and image size from each frame's P2 / image.
* ``midas_small`` / ``dpt_hybrid``: relative inverse depth sampled in the box
  (configs/depth.yaml ``sampling``), aligned per frame with ``1/Z = s*m + t`` fitted on anchor
  objects' geometric distances (``anchors``, ``alignment``). With ``leave_one_out`` an anchor's
  own MiDaS distance comes from a fit on the other anchors, so it is never scored against a
  fit that contains its own geometric estimate. Fewer than ``alignment.min_points_affine``
  anchors -> scale-only fit; none -> no estimate (coverage < 100 %).
* ``fused_<model>``: inverse-variance mean of geometric and aligned <model>, as in
  perception.fusion.DistanceFusion (no Kalman: KITTI object frames are independent).

Metrics per GT distance bin (``bins``) and over 0-60 m: n, coverage (estimates / GT objects),
AbsRel = mean |d - g| / g, RMSE, bias = mean (d - g) / g (signed; a wrong camera height or
pitch sign shows up here), median |d - g| / g. ``common`` rows use only objects that every
method that ran could estimate. Speed: per model, mean ``DepthEstimator.estimate`` time per
frame (pre + model + post at the KITTI input size) after ``fps.warmup_frames``, plus a
model-only benchmark on fixed tensors (including the live CARLA input size).

Outputs in ``out_dir``: results.json (incl. git SHA, torch/timm versions, model errors),
per_object.jsonl, depth_comparison.md, debug/*.jpg (LiDAR coloured by depth, GT boxes with GT
distances, horizon row; after the models ran, a second overlay per frame with predictions and
the relative depth maps). GT overlays are written before any model is loaded. Everything goes
to one W&B run (project aiml04-perception, group depth). The job exits non-zero if a model
failed, after writing all outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from perception.depth import CameraModel, DepthEstimator, load_camera, load_depth_config
from perception.depth.camera import resolve_path
from perception.fusion import DistanceFusion
from perception.types import Detection
from scripts.eval_lanes import find_input, git_sha, gpu_name

ALL_BIN = "all"

# ---------------------------------------------------------------- KITTI helpers (pure)

_CALIB_ALIASES = {
    "P2": ("P2",),
    "R0_rect": ("R0_rect", "R_rect"),
    "Tr_velo_to_cam": ("Tr_velo_to_cam", "Tr_velo_cam"),
}


def parse_calib(text: str) -> dict[str, np.ndarray]:
    """KITTI object calib file -> P2 (3x4), R0_rect (4x4) and Tr_velo_to_cam (4x4, padded)."""
    raw: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        arr = np.array(vals.split(), dtype=np.float64)
        if arr.size:
            raw[key.strip()] = arr

    def get(name: str) -> np.ndarray:
        for alias in _CALIB_ALIASES[name]:
            if alias in raw:
                return raw[alias]
        raise KeyError(f"calib has no {name} (keys: {sorted(raw)})")

    r0 = np.eye(4)
    r0[:3, :3] = get("R0_rect").reshape(3, 3)
    tr = np.eye(4)
    tr[:3, :4] = get("Tr_velo_to_cam").reshape(3, 4)
    return {"P2": get("P2").reshape(3, 4), "R0_rect": r0, "Tr_velo_to_cam": tr}


@dataclass
class KittiObject:
    type: str
    truncated: float
    occluded: int
    bbox: np.ndarray  # x1, y1, x2, y2 (image_2 pixels)
    dims_hwl: np.ndarray  # height, width, length (m)
    location: np.ndarray  # x, y, z of the 3D box bottom centre, rectified camera frame (y down)
    ry: float  # rotation around the camera y axis


def parse_labels(text: str) -> list[KittiObject]:
    """KITTI label_2 lines -> objects (DontCare skipped)."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 15 or f[0] == "DontCare":
            continue
        v = [float(x) for x in f[1:15]]
        out.append(
            KittiObject(
                type=f[0],
                truncated=v[0],
                occluded=int(v[1]),
                bbox=np.array(v[3:7]),
                dims_hwl=np.array(v[7:10]),
                location=np.array(v[10:13]),
                ry=v[13],
            )
        )
    return out


def load_velodyne(path: str | Path) -> np.ndarray:
    """KITTI .bin -> (N, 3) float32 x, y, z in the velodyne frame (reflectance dropped)."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)[:, :3]


def velo_to_rect(pts_velo: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 3) velodyne points -> (N, 3) rectified camera frame (x right, y down, z forward)."""
    h = np.hstack([pts_velo[:, :3].astype(np.float64), np.ones((len(pts_velo), 1))])
    return (calib["R0_rect"] @ calib["Tr_velo_to_cam"] @ h.T).T[:, :3]


def project_rect(pts_rect: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(N, 3) rectified camera points (z > 0) -> (N, 2) image pixels with projection P."""
    h = np.hstack([pts_rect, np.ones((len(pts_rect), 1))])
    uvw = (P @ h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def lidar_in_image(
    pts_velo: np.ndarray, calib: dict[str, np.ndarray], hw: tuple[int, int], min_depth: float
) -> tuple[np.ndarray, np.ndarray]:
    """(uv, pts_rect) of the points in front of the camera (z > min_depth) inside the image."""
    rect = velo_to_rect(pts_velo, calib)
    rect = rect[rect[:, 2] > min_depth]  # drop points behind / at the camera before projecting
    uv = project_rect(rect, calib["P2"])
    h, w = hw
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[ok], rect[ok]


def in_box3d(pts_rect: np.ndarray, obj: KittiObject, margin: float, clearance: float) -> np.ndarray:
    """Points inside the label's 3D box (each side enlarged by ``margin``), excluding points
    less than ``clearance`` above the box bottom (ground hits)."""
    d = pts_rect - obj.location
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    xo = c * d[:, 0] - s * d[:, 2]  # object frame: x along the length, z along the width
    zo = s * d[:, 0] + c * d[:, 2]
    yo = d[:, 1]  # 0 at the bottom, -h at the top (y down)
    h, w, length = obj.dims_hwl
    return (
        (np.abs(xo) <= length / 2 + margin)
        & (np.abs(zo) <= w / 2 + margin)
        & (yo <= -clearance)
        & (yo >= -h - margin)
    )


def gt_lidar_distance(
    uv: np.ndarray, pts_rect: np.ndarray, obj: KittiObject, gcfg: dict[str, Any]
) -> tuple[float, int]:
    """(GT distance or NaN, number of supporting LiDAR points) for one object."""
    x1, y1, x2, y2 = obj.bbox
    in2d = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    cand = pts_rect[in2d]
    sel = cand[
        in_box3d(cand, obj, float(gcfg["box3d_margin_m"]), float(gcfg["ground_clearance_m"]))
    ]
    n = int(len(sel))
    if n < int(gcfg["min_points"]):
        return float("nan"), n
    return float(np.percentile(sel[:, 2], float(gcfg["percentile"]))), n


def bin_label(d: float, bins: Sequence[Sequence[float]]) -> str | None:
    for lo, hi in bins:
        if lo <= d < hi or (d == hi and hi == bins[-1][1]):
            return f"{lo:g}-{hi:g}"
    return None


def error_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    """n, AbsRel, RMSE, signed bias (mean relative error) and median AbsRel."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if pred.size == 0:
        return {"n": 0, "absrel": None, "rmse": None, "bias": None, "medrel": None}
    rel = (pred - gt) / gt
    return {
        "n": int(pred.size),
        "absrel": float(np.mean(np.abs(rel))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "bias": float(np.mean(rel)),
        "medrel": float(np.median(np.abs(rel))),
    }


def summarize(
    records: list[dict], methods: Sequence[str], gt_key: str, bins: Sequence[Sequence[float]]
) -> dict[str, dict]:
    """{bin: {"n_gt", "methods": {m: stats + coverage}, "common": {m: stats}}}."""
    lo, hi = bins[0][0], bins[-1][1]
    groups: dict[str, list[dict]] = {f"{a:g}-{b:g}": [] for a, b in bins}
    groups[ALL_BIN] = []
    for r in records:
        g = r.get(gt_key)
        if g is None or not math.isfinite(g):
            continue
        b = bin_label(g, bins)
        if b is not None:
            groups[b].append(r)
        if lo <= g <= hi:
            groups[ALL_BIN].append(r)
    out = {}
    for name, rs in groups.items():
        entry: dict[str, Any] = {"n_gt": len(rs), "methods": {}, "common": {}}
        for m in methods:
            ok = [r for r in rs if r.get(m) is not None]
            st = error_stats([r[m] for r in ok], [r[gt_key] for r in ok])
            st["coverage"] = len(ok) / len(rs) if rs else None
            entry["methods"][m] = st
        common = [r for r in rs if all(r.get(m) is not None for m in methods)]
        entry["n_common"] = len(common)
        for m in methods:
            entry["common"][m] = error_stats([r[m] for r in common], [r[gt_key] for r in common])
        out[name] = entry
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown_table(summary: dict[str, dict], methods: Sequence[str], subset: str) -> str:
    """Rows = method x bin; ``subset`` is "methods" (each method's own coverage) or "common"."""
    head = "| method | bin | n | coverage | AbsRel | RMSE (m) | bias | median rel |"
    lines = [head, "|" + "---|" * 8]
    for m in methods:
        for b, entry in summary.items():
            st = entry[subset][m]
            cov = st.get("coverage") if subset == "methods" else None
            lines.append(
                f"| {m} | {b} | {st['n']} | {_fmt(cov, 2)} | {_fmt(st['absrel'])} | "
                f"{_fmt(st['rmse'], 2)} | {_fmt(st['bias'])} | {_fmt(st['medrel'])} |"
            )
    return "\n".join(lines)


def spread_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, min(k, n))})


# ---------------------------------------------------------------- overlays


def depth_colors(z: np.ndarray, max_depth: float) -> np.ndarray:
    """(N,) metres -> (N, 3) BGR uint8, turbo colormap (near red, far blue)."""
    v = np.clip(255.0 * (1.0 - np.asarray(z, float) / max_depth), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def _text(img: np.ndarray, text: str, org: tuple[int, int], color, dcfg: dict) -> None:
    fs, th = float(dcfg["font_scale"]), int(dcfg["text_thickness"])
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x, y = org
    y = max(tht + 2, min(img.shape[0] - 2, y))
    cv2.rectangle(img, (x, y - tht - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def draw_gt_overlay(
    img: np.ndarray,
    uv: np.ndarray,
    z: np.ndarray,
    frame: dict,
    cam: CameraModel,
    dcfg: dict,
) -> np.ndarray:
    """LiDAR points coloured by depth, horizon row, GT boxes with LiDAR / label distances."""
    out = img.copy()
    r = int(dcfg["point_radius"])
    for (u, v), c in zip(uv.astype(int), depth_colors(z, float(dcfg["max_depth_m"])), strict=True):
        cv2.circle(out, (int(u), int(v)), r, tuple(int(x) for x in c), -1)
    hz = int(round(cam.horizon_row))
    cv2.line(out, (0, hz), (out.shape[1] - 1, hz), tuple(dcfg["horizon_color"]), 1)
    th = int(dcfg["box_thickness"])
    for box in frame["skipped"]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), tuple(dcfg["skipped_color"]), 1)
    for rec in frame["records"]:
        x1, y1, x2, y2 = (int(round(v)) for v in rec["bbox"])
        ok = rec["gt_lidar"] is not None
        color = tuple(dcfg["gt_color"] if ok else dcfg["nolidar_color"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        lid = f"L{rec['gt_lidar']:.1f}" if ok else f"L- ({rec['n_lidar']}pt)"
        _text(out, f"{lid} Z{rec['gt_label_z']:.1f}", (x1, y1 - 3), color, dcfg)
    return out


def draw_pred_overlay(
    gt_overlay: np.ndarray,
    frame: dict,
    methods: Sequence[str],
    maps: dict[str, np.ndarray],
    dcfg: dict,
) -> np.ndarray:
    """GT overlay + predicted distances under each box + relative depth map strips."""
    out = gt_overlay.copy()
    short = {"geometric": "g", "midas_small": "s", "dpt_hybrid": "d"}
    for rec in frame["records"]:
        x1, _, _, y2 = (int(round(v)) for v in rec["bbox"])
        parts = []
        for m in methods:
            tag = short.get(m, "f" + short.get(m.removeprefix("fused_"), "?"))
            parts.append(f"{tag}{rec[m]:.1f}" if rec.get(m) is not None else f"{tag}-")
        if rec["anchor"]:
            parts.append("A")
        _text(out, " ".join(parts), (x1, y2 + 14), (255, 255, 255), dcfg)
    legend = "L LiDAR GT, Z label z | g geometric, s midas_small, d dpt_hybrid, f fused, A anchor"
    _text(out, legend, (4, 16), (255, 255, 255), dcfg)
    h_in = int(dcfg["inset_height"])
    strips = [out]
    for name, m in maps.items():
        lo, hi = np.nanpercentile(m, [2, 98])
        v = np.clip(255.0 * (m - lo) / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
        strip = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
        strip = cv2.resize(strip, (out.shape[1], h_in), interpolation=cv2.INTER_AREA)
        _text(strip, f"{name}: relative inverse depth (not metric)", (4, 16), (255, 255, 255), dcfg)
        strips.append(strip)
    return np.vstack(strips)


# ---------------------------------------------------------------- environment


def lib_versions() -> dict[str, str]:
    out = {}
    for mod in ("torch", "timm", "cv2", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"unavailable ({type(exc).__name__})"
    return out


def free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def model_bench(est: DepthEstimator, sizes: list[list[int]], runs: int, warmup: int) -> dict:
    """Model-only latency (backend forward on a fixed tensor, output copied to host)."""
    rng = np.random.default_rng(0)
    out = {}
    for h, w in sizes:
        blob = rng.standard_normal((1, 3, int(h), int(w))).astype(np.float32)
        for _ in range(warmup):
            est.backend(blob)
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            est.backend(blob)
            t.append(1e3 * (time.perf_counter() - t0))
        out[f"{h}x{w}"] = {
            "mean_ms": float(np.mean(t)),
            "p50_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)),
            "fps": 1e3 / float(np.mean(t)),
        }
    return out


# ---------------------------------------------------------------- main


def resolve_data_dirs(cfg: dict) -> dict[str, Path]:
    dcfg = yaml.safe_load(resolve_path(cfg["data_config"]).read_text(encoding="utf-8"))
    root, obj = dcfg["input_root"], dcfg["object"]
    dirs = {}
    for key in ("image_dir", "calib_dir", "label_dir", "velodyne_dir"):
        primary = str(Path(root, obj[key]))
        dirs[key] = find_input(primary, cfg["search_root"], int(cfg["search_depth"]), 2)
        if not dirs[key].is_dir():
            raise FileNotFoundError(f"KITTI {key} is not a directory: {dirs[key]}")
    return dirs


def build_frame(
    fid: str,
    dirs: dict[str, Path],
    cfg: dict,
    base_cam: CameraModel,
    fusion: DistanceFusion,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """GT + geometric estimates for one frame. Returns (frame, image, uv, lidar z)."""
    img = cv2.imread(str(dirs["image_dir"] / f"{fid}.png"))
    if img is None:
        raise FileNotFoundError(f"cannot read image {fid}")
    hw = (int(img.shape[0]), int(img.shape[1]))
    calib = parse_calib((dirs["calib_dir"] / f"{fid}.txt").read_text())
    labels = parse_labels((dirs["label_dir"] / f"{fid}.txt").read_text())
    pts = load_velodyne(dirs["velodyne_dir"] / f"{fid}.bin")
    gcfg, ocfg = cfg["gt"], cfg["objects"]
    uv, rect = lidar_in_image(pts, calib, hw, float(gcfg["min_depth_m"]))
    cam = base_cam.with_intrinsics(calib["P2"], hw)
    cmap = ocfg["class_map"]
    names = sorted(set(cmap.values()))
    keep, skipped = [], []
    for o in labels:
        h = o.bbox[3] - o.bbox[1]
        if (
            o.type in cmap
            and h >= float(ocfg["min_box_h_px"])
            and o.occluded <= int(ocfg["max_occluded"])
            and o.truncated <= float(ocfg["max_truncated"])
        ):
            keep.append(o)
        else:
            skipped.append(o.bbox.tolist())
    dets = [
        Detection(o.bbox.astype(np.float32), 1.0, names.index(cmap[o.type]), cmap[o.type])
        for o in keep
    ]
    meas, _ = fusion.measure(dets, hw, None, camera=cam)
    records = []
    for i, (o, ms) in enumerate(zip(keep, meas, strict=True)):
        gl, n = gt_lidar_distance(uv, rect, o, gcfg)
        records.append(
            {
                "frame": fid,
                "obj": i,
                "type": o.type,
                "cls": cmap[o.type],
                "bbox": [float(v) for v in o.bbox],
                "occluded": o.occluded,
                "truncated": o.truncated,
                "n_lidar": n,
                "gt_lidar": gl if math.isfinite(gl) else None,
                "gt_label_z": float(o.location[2]),
                "geometric": ms.geometric[0] if ms.geometric else None,
                "geometric_std": ms.geometric[1] if ms.geometric else None,
                "anchor": False,
            }
        )
    frame = {"id": fid, "hw": hw, "P2": calib["P2"], "dets": dets, "records": records}
    frame["skipped"] = skipped
    return frame, img, uv, rect[:, 2]


def run_model(
    name: str,
    frames: list[dict],
    dirs: dict[str, Path],
    cfg: dict,
    dcfg_depth: dict,
    base_cam: CameraModel,
    fusion: DistanceFusion,
    debug_ids: set[str],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one depth model over all frames, filling ``<name>`` / ``fused_<name>`` in records."""
    mcfg = copy.deepcopy(dcfg_depth)
    mcfg["estimator"] = {**mcfg["estimator"], **cfg["estimator"], "model": name}
    mcfg["models"][name] = {**mcfg["models"][name], **cfg["model_overrides"].get(name, {})}
    t_load = time.perf_counter()
    est = DepthEstimator(mcfg)
    info: dict[str, Any] = {
        "input_hw": mcfg["models"][name]["input_hw"],
        "load_s": time.perf_counter() - t_load,
    }
    timing: dict[str, list[float]] = defaultdict(list)
    modes: dict[str, int] = defaultdict(int)
    maps: dict[str, np.ndarray] = {}
    warm = min(int(cfg["fps"]["warmup_frames"]), len(frames) - 1)
    loo = bool(cfg["leave_one_out"])
    for k, fr in enumerate(frames):
        img = cv2.imread(str(dirs["image_dir"] / f"{fr['id']}.png"))
        inv = est.estimate(img)
        if k >= warm:
            for key, v in est.last_timing_ms.items():
                timing[key].append(v)
        cam = base_cam.with_intrinsics(fr["P2"], fr["hw"])
        meas, align = fusion.measure(fr["dets"], fr["hw"], inv, camera=cam, leave_one_out=loo)
        modes[align.mode if align else "none"] += 1
        for rec, ms in zip(fr["records"], meas, strict=True):
            rec["anchor"] = ms.anchor
            rec[f"{name}_raw"] = ms.midas_value if math.isfinite(ms.midas_value) else None
            rec[name] = ms.midas[0] if ms.midas else None
            rec[f"fused_{name}"] = ms.fused[0] if ms.fused else None
        if fr["id"] in debug_ids:
            small = cv2.resize(inv, (inv.shape[1] // 2, inv.shape[0] // 2))
            maps[fr["id"]] = small
    info["frame_alignment_modes"] = dict(modes)
    info["timing_ms"] = {k: float(np.mean(v)) for k, v in timing.items() if v}
    if "total_ms" in info["timing_ms"]:
        info["fps"] = 1e3 / info["timing_ms"]["total_ms"]
    b = cfg["fps"]["model_bench"]
    info["model_bench"] = model_bench(
        est, b["sizes"].get(name, []), int(b["runs"]), int(b["warmup"])
    )
    del est
    free_gpu()
    return info, maps


def no_lidar_counts(records: list[dict], bins: Sequence[Sequence[float]]) -> dict:
    miss = [r for r in records if r["gt_lidar"] is None]
    by_bin: dict[str, int] = defaultdict(int)
    for r in miss:
        by_bin[bin_label(r["gt_label_z"], bins) or "other"] += 1
    return {
        "total_objects": len(records),
        "no_lidar": len(miss),
        "by_label_z_bin": dict(by_bin),
        "by_class": dict(_count(miss, "cls")),
        "by_occluded": {str(k): v for k, v in _count(miss, "occluded").items()},
        "zero_points": sum(r["n_lidar"] == 0 for r in miss),
    }


def _count(rs: list[dict], key: str) -> dict:
    out: dict = defaultdict(int)
    for r in rs:
        out[r[key]] += 1
    return out


def lidar_vs_label(records: list[dict]) -> dict:
    """Median label z - LiDAR z per class (expect ~ half the object length, > 0)."""
    out = {}
    for cls in sorted({r["cls"] for r in records}):
        d = [r["gt_label_z"] - r["gt_lidar"] for r in records if r["cls"] == cls and r["gt_lidar"]]
        if d:
            out[cls] = {
                "n": len(d),
                "median_m": float(np.median(d)),
                "p90_m": float(np.percentile(d, 90)),
            }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/depth_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    dcfg_depth = load_depth_config(cfg["depth_config"])
    base_cam = load_camera(cfg["camera"]["config"], cfg["camera"]["name"])
    fusion = DistanceFusion(dcfg_depth, base_cam)
    dbg = cfg["debug"]
    n_frames = int(cfg["smoke"]["n_frames"] if args.smoke else cfg["sample"]["n_frames"])
    debug_on = bool(dbg["enabled"]) or (args.smoke and bool(cfg["smoke"]["debug"]))
    out_dir = Path(cfg["out_dir"]) / ("smoke" if args.smoke else "full")
    debug_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = resolve_data_dirs(cfg)
    ids = sorted(p.stem for p in dirs["image_dir"].glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"no png images in {dirs['image_dir']}")
    ids = sorted(random.Random(int(cfg["sample"]["seed"])).sample(ids, min(n_frames, len(ids))))
    debug_ids = (
        {ids[i] for i in spread_indices(len(ids), int(dbg["n_images"]))} if debug_on else set()
    )
    info: dict[str, Any] = {
        "git_sha": git_sha(),
        "gpu": gpu_name(),
        "versions": lib_versions(),
        "n_frames": len(ids),
        "smoke": args.smoke,
        "data_dirs": {k: str(v) for k, v in dirs.items()},
        "camera": cfg["camera"],
        "gt_primary": cfg["gt"]["primary"],
        "boxes": "KITTI GT 2D boxes (score 1), not detector output",
        "alignment": "per frame 1/Z = s*m + t on anchors' geometric distances"
        + (", leave-one-out for anchors" if cfg["leave_one_out"] else ""),
    }
    print(json.dumps(info, indent=1), flush=True)

    import wandb

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"],
        group=w["group"],
        job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""),
        config={**cfg, "depth": dcfg_depth, **info},
    )

    # Phase 1: GT and geometric (no model loaded yet), GT overlays first.
    frames: list[dict] = []
    gt_overlays: dict[str, np.ndarray] = {}
    if debug_on:
        debug_dir.mkdir(parents=True, exist_ok=True)
    jq = [int(cv2.IMWRITE_JPEG_QUALITY), int(dbg["jpeg_quality"])]
    t0 = time.time()
    for fid in ids:
        fr, img, uv, z = build_frame(fid, dirs, cfg, base_cam, fusion)
        frames.append(fr)
        if fid in debug_ids:
            cam = base_cam.with_intrinsics(fr["P2"], fr["hw"])
            ov = draw_gt_overlay(img, uv, z, fr, cam, dbg)
            gt_overlays[fid] = ov
            cv2.imwrite(str(debug_dir / f"{fid}_gt.jpg"), ov, jq)
    records = [r for fr in frames for r in fr["records"]]
    info["gt_phase_s"] = time.time() - t0
    info["no_lidar"] = no_lidar_counts(records, cfg["bins"])
    info["lidar_vs_label"] = lidar_vs_label(records)
    print("no LiDAR support:", info["no_lidar"], flush=True)
    print("label z - LiDAR z:", info["lidar_vs_label"], flush=True)
    if gt_overlays:
        run.log(
            {
                "debug_gt": [
                    wandb.Image(str(debug_dir / f"{f}_gt.jpg"), caption=f)
                    for f in sorted(gt_overlays)
                ]
            }
        )

    # Phase 2: depth models, each isolated so one failure does not lose the others.
    model_info: dict[str, dict] = {}
    model_maps: dict[str, dict[str, np.ndarray]] = {}
    failed: list[str] = []
    for name in cfg["models"]:
        print(f"--- {name}", flush=True)
        try:
            mi, maps = run_model(name, frames, dirs, cfg, dcfg_depth, base_cam, fusion, debug_ids)
            model_info[name] = mi
            model_maps[name] = maps
            print(name, json.dumps({k: v for k, v in mi.items()}, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded and reported, job fails at the end
            failed.append(name)
            model_info[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(traceback.format_exc(), flush=True)
            for r in records:  # no partial results from a failed model
                for key in (name, f"fused_{name}", f"{name}_raw"):
                    r.pop(key, None)
            free_gpu()
    info["models"] = model_info

    ran = [m for m in cfg["models"] if m not in failed]
    methods = ["geometric", *ran, *(f"fused_{m}" for m in ran)]
    bins = cfg["bins"]
    results = {
        "setup": info,
        "methods": methods,
        "primary": summarize(records, methods, "gt_lidar", bins),
        "label_z": summarize(records, methods, "gt_label_z", bins),
        "beyond_last_bin": sum(1 for r in records if (r["gt_lidar"] or 0) > bins[-1][1]),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    with open(out_dir / "per_object.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    fps_lines = ["| model | input | estimate ms (pre+model+post) | FPS | model-only ms by input |"]
    fps_lines.append("|---|---|---|---|---|")
    for m in ran:
        mi = model_info[m]
        bench = ", ".join(f"{k}: {v['mean_ms']:.1f}" for k, v in mi["model_bench"].items())
        fps_lines.append(
            f"| {m} | {mi['input_hw']} | {_fmt(mi['timing_ms'].get('total_ms'), 1)} | "
            f"{_fmt(mi.get('fps'), 1)} | {bench} |"
        )
    md = [
        f"# M4.4 distance vs KITTI ({len(ids)} frames, git {info['git_sha'][:8]}, {info['gpu']})",
        "",
        f"Boxes: {info['boxes']}. MiDaS: {info['alignment']}.",
        "",
        "## GT = LiDAR (primary), each method on the objects it covers",
        markdown_table(results["primary"], methods, "methods"),
        "",
        "## GT = LiDAR, common subset (objects every method estimated)",
        markdown_table(results["primary"], methods, "common"),
        "",
        "## GT = label_2 location z (secondary, box centre: expect negative bias)",
        markdown_table(results["label_z"], methods, "methods"),
        "",
        f"## Speed ({info['gpu']})",
        *fps_lines,
        "",
        f"No LiDAR support: {json.dumps(info['no_lidar'])}",
        f"Label z - LiDAR z by class: {json.dumps(info['lidar_vs_label'])}",
        f"Failed models: {failed or 'none'}",
    ]
    (out_dir / "depth_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)

    pred_files = []
    for fr in frames:
        if fr["id"] not in gt_overlays:
            continue
        maps = {
            m: cv2.resize(mm[fr["id"]], (fr["hw"][1], fr["hw"][0]))
            for m, mm in model_maps.items()
            if fr["id"] in mm
        }
        ov = draw_pred_overlay(gt_overlays[fr["id"]], fr, methods, maps, dbg)
        p = debug_dir / f"{fr['id']}_pred.jpg"
        cv2.imwrite(str(p), ov, jq)
        pred_files.append(p)

    rows = []
    for gt_name in ("primary", "label_z"):
        for b, entry in results[gt_name].items():
            for m in methods:
                st = entry["methods"][m]
                rows.append(
                    [
                        gt_name,
                        b,
                        m,
                        st["n"],
                        st.get("coverage"),
                        st["absrel"],
                        st["rmse"],
                        st["bias"],
                    ]
                )
    log: dict[str, Any] = {
        "comparison": wandb.Table(
            columns=["gt", "bin", "method", "n", "coverage", "absrel", "rmse", "bias"], data=rows
        )
    }
    if pred_files:
        log["debug_pred"] = [wandb.Image(str(p), caption=p.stem) for p in pred_files]
    run.log(log)
    for m in methods:
        st = results["primary"][ALL_BIN]["methods"][m]
        run.summary[f"{m}/absrel"] = st["absrel"]
        run.summary[f"{m}/rmse"] = st["rmse"]
        run.summary[f"{m}/coverage"] = st["coverage"]
    for m in ran:
        run.summary[f"{m}/fps"] = model_info[m].get("fps")
    run.summary["no_lidar"] = info["no_lidar"]["no_lidar"]
    run.summary["failed_models"] = ",".join(failed)
    run.finish()
    if failed:
        raise SystemExit(f"depth models failed: {failed} (see results.json setup.models)")


if __name__ == "__main__":
    main()
