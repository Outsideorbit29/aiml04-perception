"""DistanceFusion: tracks + relative depth + lanes -> PerceptionFrame (M4.3, M4.5).

    fusion = DistanceFusion(load_depth_config("configs/depth.yaml"),
                            load_camera("configs/camera.yaml", "carla"))
    inv = est.maybe_estimate(img)                       # None on skipped frames
    frame = fusion.update(tracks, lanes, stamp, frame_id, img.shape[:2], inv_depth=inv)

Per object (only classes in ``fusion.object_classes``; traffic lights/signs have no ground
contact and would otherwise become false lead objects):
1. Geometric distance from the box bottom row (perception.depth.camera, flat ground). Invalid
   when the box touches the image bottom (contact point not visible), the row is not far enough
   below the horizon, or the distance exceeds ``geometry.max_distance_m``.
2. MiDaS value in the lower-central part of the box (``sampling``), if a depth map is given.
3. Per-frame alignment ``1/Z = s*m + t`` on anchors = confident, near objects with a valid
   geometric distance (``anchors``). If no fit is possible, the last fit is reused for up to
   ``fusion.reuse_alignment_frames`` frames; MiDaS never yields a distance without a fit.
4. Inverse-variance fusion of the geometric and aligned-MiDaS distances. The aligned-MiDaS
   std is never below the camera's geometric inverse-depth noise (the anchors' own error), and
   exactly determined fits are inflated (perception.depth.Alignment.inv_std).
5. Per-track Kalman filter (perception.fusion.kalman) -> distance, std, closing speed, TTC.
6. ``in_ego_lane`` = ``LaneResult.contains(box bottom-centre)``.
Objects with no distance at all (no geometric, no aligned MiDaS, no live filter) are left out
of the frame, because ``ObjectState.distance_m`` must be a real number; ``last_stats`` counts
them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.depth.alignment import Alignment, ScaleAligner
from perception.depth.camera import CameraModel
from perception.depth.estimator import sample_box
from perception.fusion.kalman import FilterBank
from perception.types import Detection, LaneResult, ObjectState, PerceptionFrame, Track

Estimate = tuple[float, float]  # (distance m, 1-sigma m)


@dataclass
class Measurement:
    """Single-frame distance estimates for one detection (before temporal filtering)."""

    det: Detection
    geometric: Estimate | None
    midas_value: float  # raw relative inverse depth (NaN without a depth map)
    midas: Estimate | None  # aligned MiDaS distance
    fused: Estimate | None
    anchor: bool  # used to fit the alignment
    align_mode: str | None = None  # mode of the fit that produced ``midas`` (affine | scale)
    align_anchors: int = 0  # anchors in that fit


def fuse(estimates: list[Estimate | None]) -> Estimate | None:
    """Inverse-variance weighted mean of the available estimates."""
    valid = [e for e in estimates if e is not None and math.isfinite(e[0]) and e[1] > 0]
    if not valid:
        return None
    w = np.array([1.0 / (s * s) for _, s in valid])
    z = np.array([d for d, _ in valid])
    return float((w * z).sum() / w.sum()), float(1.0 / math.sqrt(w.sum()))


class DistanceFusion:
    def __init__(self, cfg: dict[str, Any], camera: CameraModel) -> None:
        self.cfg = cfg
        self.camera = camera
        g = cfg["geometry"]
        self.min_angle = math.radians(float(g["min_angle_deg"]))
        self.bottom_margin = float(g["bottom_margin_px"])
        self.max_geo = float(g["max_distance_m"])
        self.sampling = cfg["sampling"]
        a = cfg["anchors"]
        self.anchor_min_score = float(a["min_score"])
        self.anchor_max_m = float(a["max_distance_m"])
        self.anchor_min_h = float(a["min_box_h_px"])
        f = cfg["fusion"]
        self.classes = set(f["object_classes"])
        self.reuse_frames = int(f["reuse_alignment_frames"])
        self.aligner = ScaleAligner(cfg["alignment"])
        self.bank = FilterBank(cfg["kalman"])
        self.last_alignment: Alignment | None = None
        self._align_age = 0
        self.last_stats: dict[str, int] = {}
        self.last_measurements: list[Measurement] = []

    def reset(self) -> None:
        """Forget filters and the last alignment (call between unrelated clips)."""
        self.bank.reset()
        self.last_alignment = None
        self._align_age = 0

    def keeps(self, det: Detection) -> bool:
        return det.cls_name in self.classes

    # ------------------------------------------------------------------ single frame
    def geometric(self, det: Detection, cam: CameraModel) -> Estimate | None:
        _, y2 = det.bottom_center
        if y2 > cam.height - 1 - self.bottom_margin:
            return None  # box cut by the image bottom: the ground contact is not visible
        est = cam.ground_distance(y2, self.min_angle)
        if est is None or est[0] > self.max_geo:
            return None
        return est

    def is_anchor(self, det: Detection, geo: Estimate | None, m: float) -> bool:
        x1, y1, x2, y2 = (float(v) for v in det.bbox_xyxy)
        return (
            geo is not None
            and math.isfinite(m)
            and det.score >= self.anchor_min_score
            and geo[0] <= self.anchor_max_m
            and (y2 - y1) >= self.anchor_min_h
        )

    def measure(
        self,
        dets: list[Detection],
        hw: tuple[int, int],
        inv_depth: np.ndarray | None = None,
        camera: CameraModel | None = None,
        leave_one_out: bool = False,
        alignment_fallback: Alignment | None = None,
    ) -> tuple[list[Measurement], Alignment | None]:
        """Stateless per-frame estimates for ``dets`` (all of them; filter with ``keeps``).

        ``camera`` overrides the configured camera (e.g. per-frame KITTI calib); it is rescaled
        to ``hw``. ``leave_one_out`` (evaluation only) aligns each anchor's MiDaS distance with a
        fit that excludes that anchor, so it is not scored against its own geometric estimate.
        ``alignment_fallback`` is used when this frame's anchors give no fit.
        Returns the measurements and the alignment fitted on all anchors (or the fallback).
        """
        cam = (camera or self.camera).scaled_to(hw)
        geo = [self.geometric(d, cam) for d in dets]
        if inv_depth is not None:
            if inv_depth.shape[:2] != tuple(hw):
                raise ValueError(f"depth map {inv_depth.shape[:2]} does not match frame {hw}")
            m = np.array([sample_box(inv_depth, d.bbox_xyxy, self.sampling) for d in dets])
        else:
            m = np.full(len(dets), np.nan)
        anchor = np.array(
            [self.is_anchor(d, g, v) for d, g, v in zip(dets, geo, m, strict=True)], bool
        )
        z_geo = np.array([g[0] if g is not None else np.nan for g in geo])
        align = None
        if inv_depth is not None:
            align = self.aligner.fit(m[anchor], z_geo[anchor]) or alignment_fallback
        out = []
        for i, d in enumerate(dets):
            a = align
            if leave_one_out and anchor[i]:
                keep = anchor.copy()
                keep[i] = False
                a = self.aligner.fit(m[keep], z_geo[keep])
            midas = None
            if a is not None and math.isfinite(m[i]):
                z = float(a.to_depth(m[i]))
                if math.isfinite(z):
                    midas = (z, float(a.depth_std(z, cam.inv_depth_std())))
            out.append(
                Measurement(
                    d,
                    geo[i],
                    float(m[i]),
                    midas,
                    fuse([geo[i], midas]),
                    bool(anchor[i]),
                    a.mode if midas is not None else None,
                    a.n_anchors if midas is not None else 0,
                )
            )
        return out, align

    # ------------------------------------------------------------------ video
    def update(
        self,
        tracks: list[Track],
        lanes: LaneResult | None,
        stamp: float,
        frame_id: int,
        hw: tuple[int, int],
        inv_depth: np.ndarray | None = None,
        timings_ms: dict[str, float] | None = None,
    ) -> PerceptionFrame:
        """One video frame: ``tracks`` from the Tracker, ``lanes`` from the lane detector,
        ``inv_depth`` from ``DepthEstimator.maybe_estimate`` (None on skipped frames)."""
        t0 = time.perf_counter()
        tracks = [t for t in tracks if self.keeps(t.det)]
        fallback = None
        if self.last_alignment is not None and self._align_age < self.reuse_frames:
            fallback = self.last_alignment
        meas, align = self.measure(
            [t.det for t in tracks], hw, inv_depth, alignment_fallback=fallback
        )
        if inv_depth is not None and align is not None and align is not fallback:
            self.last_alignment, self._align_age = align, 0
        else:
            self._align_age += 1
        lanes = lanes if lanes is not None else LaneResult(None, None, None, 0.0)
        objects: list[ObjectState] = []
        dropped = 0
        for tr, ms in zip(tracks, meas, strict=True):
            if ms.fused is not None:
                est = self.bank.update(tr.track_id, tr.det.cls, stamp, *ms.fused)
            else:
                est = self.bank.predict_only(tr.track_id, stamp)
            if est is None:
                dropped += 1
                continue
            objects.append(
                ObjectState(
                    track=tr,
                    distance_m=est.distance_m,
                    distance_std=est.distance_std,
                    closing_speed_mps=est.closing_speed_mps,
                    ttc_s=est.ttc_s,
                    in_ego_lane=bool(lanes.contains(*tr.det.bottom_center)),
                )
            )
        self.bank.prune(stamp)
        self.last_measurements = meas
        self.last_stats = {
            "objects": len(objects),
            "no_estimate": dropped,
            "anchors": int(sum(m.anchor for m in meas)),
            "aligned": int(align is not None),
            "filters": len(self.bank.filters),
        }
        timings = dict(timings_ms or {})
        timings["fusion_ms"] = 1e3 * (time.perf_counter() - t0)
        return PerceptionFrame(stamp, frame_id, objects, lanes, timings)
