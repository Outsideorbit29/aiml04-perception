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
4. Inverse-variance fusion of the geometric and aligned-MiDaS estimates in INVERSE depth
   (``fuse_inverse``), where both errors are roughly Gaussian with a std that does not depend
   on the distance (CameraModel.inv_depth_std, perception.depth.Alignment.inv_std). Fusing
   distances with stds Z^2 * k evaluated at each estimate's own Z would give the shorter
   estimate more weight and bias the result short (-10 % at 45 m with KITTI noise). The
   aligned-MiDaS std is never below the camera's geometric inverse-depth noise (the anchors'
   own error), and exactly determined fits are inflated. The two estimates are NOT independent:
   MiDaS is fitted to the geometric distances of the anchors, so it inherits their shared
   error (mostly the pitch error). The fused std therefore assumes a correlation
   ``fusion.correlation`` between them (the weights, and so the fused distance, do not depend
   on it); 0 would be the independent-errors std, 1 gives no gain over the better estimate.
   std(Z) = Z^2 * std(1/Z).
5. Per-track Kalman filter (perception.fusion.kalman) -> distance, std, closing speed, TTC.
6. ``in_ego_lane`` (``ego_lane``): lane check plus a metric lateral corridor gate. The lateral
   offset of the box bottom-centre is ``X = (u - cx) * Zc / fx`` (``Zc`` = camera-frame depth
   of that ground point, = Z at zero pitch). The box centre (not its inner edge) is used: a
   kerb-parallel parked car's box spans from its near outer corner to its far inner corner, so
   its inner edge times the near distance badly underestimates the offset, while the centre
   stays ~2.2-2.5 m for a car parked 2.9 m out. Z for the offset is the larger of the filtered
   distance and this frame's measurement: a too-short Z shrinks |X| towards 0 for any box, while
   a longer Z cannot push a car straight ahead (u ~ cx) out of the corridor.
   - Both boundaries measured (``left`` and ``right``) and ``confidence >= min_lane_confidence``:
     ``contains(bottom-centre)`` AND ``|X| <= half_width_lanes_m + Z^2 / (2 curve_radius_m)``
     at every range. The second term is the camera-frame drift of a lane curving with radius
     ``curve_radius_m``; without it the gate would drop leads on curves, and switching the gate
     off at range let YOLOP's ego area (which covers parked cars on unmarked streets) decide.
   - Otherwise (one side synthesized, low confidence, or no lane result): corridor only,
     ``|X| <= half_width_no_lanes_m`` within ``range_no_lanes_m``, else False. The synthesized
     mask can spill over the kerb, so it is not used; a lead car on an unmarked street is still
     found. No curve allowance here: without lane evidence a parked car would get in at range.
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


def fuse_inverse(estimates: list[Estimate | None], correlation: float = 0.0) -> Estimate | None:
    """Fuse inverse-depth estimates ``(1/Z, std of 1/Z)`` -> distance estimate ``(Z, std Z)``.

    Inverse-variance weighted mean in inverse depth. The returned std is the std of that
    weighted mean when every pair of estimates has error correlation ``correlation`` (in [0, 1];
    0 = independent), propagated with std(Z) = Z^2 std(1/Z). A single valid estimate is
    returned as is.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError(f"correlation {correlation} not in [0, 1]")
    valid = [
        e
        for e in estimates
        if e is not None and math.isfinite(e[0]) and e[0] > 0 and math.isfinite(e[1]) and e[1] > 0
    ]
    if not valid:
        return None
    u = np.array([v for v, _ in valid])
    sd = np.array([s for _, s in valid])
    a = 1.0 / (sd * sd)
    a /= a.sum()
    inv = float((a * u).sum())
    corr = np.full((len(valid), len(valid)), correlation)
    np.fill_diagonal(corr, 1.0)
    asd = a * sd
    inv_std = math.sqrt(max(float(asd @ corr @ asd), 0.0))
    z = 1.0 / inv
    return z, z * z * inv_std


def fuse(estimates: list[Estimate | None], correlation: float = 0.0) -> Estimate | None:
    """Fuse distance estimates ``(Z, std Z)`` whose error is Gaussian in inverse depth
    (std(1/Z) = std(Z) / Z^2, as for flat-ground geometry and aligned MiDaS): converts them and
    calls ``fuse_inverse``. A single valid estimate is returned as is."""
    valid = [
        e
        for e in estimates
        if e is not None and math.isfinite(e[0]) and e[0] > 0 and math.isfinite(e[1]) and e[1] > 0
    ]
    if len(valid) == 1:
        return valid[0]
    return fuse_inverse([(1.0 / z, s / (z * z)) for z, s in valid], correlation)


@dataclass(frozen=True)
class EgoCorridor:
    """``fusion.ego_lane`` settings (configs/depth.yaml), see the module docstring item 6."""

    min_lane_confidence: float
    half_width_lanes_m: float
    curve_radius_m: float
    half_width_no_lanes_m: float
    range_no_lanes_m: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EgoCorridor:
        out = cls(**{k: float(d[k]) for k in cls.__dataclass_fields__})
        for k, v in vars(out).items():
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"fusion.ego_lane.{k} = {v} must be finite and >= 0")
        if out.curve_radius_m <= 0:
            raise ValueError("fusion.ego_lane.curve_radius_m must be > 0")
        return out

    def half_width_lanes(self, z: float) -> float:
        """Corridor half-width with trusted lanes at distance z (curve allowance included)."""
        return self.half_width_lanes_m + z * z / (2.0 * self.curve_radius_m)


def lateral_offset(u: float, z: float, cam: CameraModel) -> float:
    """Lateral offset (m, + = right) of the ground point at image column ``u`` and forward
    distance ``z`` (flat ground, camera pitched down by ``cam.pitch_rad``)."""
    zc = z * math.cos(cam.pitch_rad) + cam.height_m * math.sin(cam.pitch_rad)
    return (u - cam.cx) * zc / cam.fx


def two_sided(lanes: LaneResult | None, min_confidence: float) -> bool:
    """Both lane boundaries measured (not synthesized) with enough confidence."""
    return (
        lanes is not None
        and lanes.left is not None
        and lanes.right is not None
        and lanes.confidence >= min_confidence
    )


def in_ego_lane(
    det: Detection,
    z: float,
    lanes: LaneResult | None,
    cam: CameraModel,
    corridor: EgoCorridor,
    z_lateral: float | None = None,
) -> bool:
    """Ego-lane flag of an object at filtered distance ``z`` (module docstring item 6).
    ``z_lateral`` (default ``z``) is the distance used for the lateral offset; ``update`` passes
    max(filtered, measured). ``cam`` must already be scaled to the frame size."""
    zl = z if z_lateral is None else max(z, z_lateral)
    if not (math.isfinite(z) and z >= 0 and math.isfinite(zl)):
        return False
    u, v = det.bottom_center
    lat = abs(lateral_offset(u, zl, cam))
    if two_sided(lanes, corridor.min_lane_confidence):
        return lat <= corridor.half_width_lanes(zl) and bool(lanes.contains(u, v))
    return z <= corridor.range_no_lanes_m and lat <= corridor.half_width_no_lanes_m


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
        self.correlation = float(f["correlation"])
        self.corridor = EgoCorridor.from_dict(f["ego_lane"])
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError(f"fusion.correlation {self.correlation} not in [0, 1]")
        if not camera.inv_depth_std() > 0:
            raise ValueError(
                "camera sigma_v_px and sigma_pitch_deg are both 0: the geometric distance would "
                "have zero std and could not be fused (set them in configs/camera.yaml)"
            )
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
        geo_inv_std = cam.inv_depth_std()
        if not geo_inv_std > 0:
            raise ValueError(
                "camera has zero geometric inverse-depth std (sigma_v_px, sigma_pitch)"
            )
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
            inv_midas = None
            if a is not None and math.isfinite(m[i]):
                z = float(a.to_depth(m[i]))
                if math.isfinite(z):
                    inv_s = a.inv_std(geo_inv_std)
                    midas = (z, z * z * inv_s)
                    inv_midas = (1.0 / z, inv_s)
            inv_geo = (1.0 / geo[i][0], geo_inv_std) if geo[i] is not None else None
            out.append(
                Measurement(
                    d,
                    geo[i],
                    float(m[i]),
                    midas,
                    fuse_inverse([inv_geo, inv_midas], self.correlation),
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
        cam = self.camera.scaled_to(hw)
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
                    in_ego_lane=in_ego_lane(
                        tr.det,
                        est.distance_m,
                        lanes,
                        cam,
                        self.corridor,
                        z_lateral=ms.fused[0] if ms.fused is not None else None,
                    ),
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
