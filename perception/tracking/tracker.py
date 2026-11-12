"""Multi-object tracker (ARCHITECTURE.md section 3.3, PLAN.md M2.5).

    tracker = Tracker(load_config("configs/tracker.yaml"))
    tracks = tracker.update(detections, frame)   # -> list[perception.types.Track]

ByteTrack comes from ``supervision`` (numpy + scipy, no torch), so the same class runs on Kaggle
and in the laptop ROS2 runtime, where PyTorch is not installed. Ultralytics' BYTETracker would
pull in torch. ``supervision`` assigns the ids; this wrapper adds the ``Track`` age and box
history that the data contract asks for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.detection.processing import box_iou
from perception.types import Detection, Track

REQUIRED_KEYS = (
    "type",
    "frame_rate",
    "track_activation_threshold",
    "lost_track_buffer",
    "minimum_matching_threshold",
    "minimum_consecutive_frames",
    "history_len",
)
REFERENCE_FPS = 30.0  # ByteTrack defines lost_track_buffer at 30 fps and rescales by frame_rate


@dataclass
class _TrackState:
    first_frame: int
    last_frame: int  # last frame this id was output (ByteTrack may update it without output)
    history: deque = field(default_factory=deque)


class Tracker:
    """``Tracker(cfg).update(dets, frame) -> list[Track]`` for the tracks matched in this frame.

    ``Track.age`` is the number of frames since the track was first output (0 on its first
    frame). ``Track.history`` holds the last ``history_len`` boxes (xyxy) of that track.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"tracker config is missing {missing}")
        self.cfg = dict(cfg)
        kind = cfg["type"]
        if kind == "deepsort":
            raise NotImplementedError("DeepSORT is the M2.7 stretch goal")
        if kind != "bytetrack":
            raise ValueError(f"unknown tracker type {kind!r}; choose bytetrack")
        import supervision as sv  # lazy: keeps `import perception.tracking` cheap

        self._sv = sv
        self.history_len = int(cfg["history_len"])
        # ByteTrack drops a track this many frames after its last *update*. supervision outputs a
        # matched track only if its Kalman box overlaps the detection (IoU > 0.5), so the last
        # *output* can be older than the last update. Without access to ByteTrack's live tracks
        # we keep states for 2 * max_lost frames since the last output instead.
        self.max_lost = max(1, int(cfg["frame_rate"] / REFERENCE_FPS * cfg["lost_track_buffer"]))
        self.reset()

    def reset(self) -> None:
        """Forget all tracks (call between unrelated videos)."""
        c = self.cfg
        self._bt = self._sv.ByteTrack(
            track_activation_threshold=float(c["track_activation_threshold"]),
            lost_track_buffer=int(c["lost_track_buffer"]),
            minimum_matching_threshold=float(c["minimum_matching_threshold"]),
            frame_rate=int(c["frame_rate"]),
            minimum_consecutive_frames=int(c["minimum_consecutive_frames"]),
        )
        self.frame_idx = 0
        self._states: dict[int, _TrackState] = {}

    def _to_sv(self, dets: list[Detection]) -> Any:
        if not dets:
            return self._sv.Detections.empty()
        return self._sv.Detections(
            xyxy=np.stack([np.asarray(d.bbox_xyxy, np.float32) for d in dets]),
            confidence=np.array([d.score for d in dets], np.float32),
            class_id=np.array([d.cls for d in dets], int),
            data={"idx": np.arange(len(dets))},
        )

    def _live_ids(self) -> set[int] | None:
        """External ids ByteTrack still holds (tracked or lost), or None if not exposed."""
        pools = [getattr(self._bt, n, None) for n in ("tracked_tracks", "lost_tracks")]
        if any(p is None for p in pools):
            return None
        try:
            return {int(t.external_track_id) for pool in pools for t in pool}
        except (AttributeError, TypeError, ValueError):
            return None

    def _prune(self) -> None:
        """Drop per-id state only once ByteTrack itself can no longer return that id."""
        live = self._live_ids()
        if live is not None:
            dead = [t for t in self._states if t not in live]
        else:
            cutoff = self.frame_idx - 2 * self.max_lost
            dead = [t for t, s in self._states.items() if s.last_frame < cutoff]
        for t in dead:
            del self._states[t]

    @staticmethod
    def _source_rows(out: Any, dets: list[Detection]) -> list[int]:
        """Index into ``dets`` of each output row (via the ``idx`` data column, else by IoU)."""
        n = len(out)
        if n == 0:
            return []
        data = getattr(out, "data", None) or {}
        if "idx" in data:
            return [int(i) for i in data["idx"]]
        boxes = np.stack([d.bbox_xyxy for d in dets])
        return [int(i) for i in box_iou(np.asarray(out.xyxy), boxes).argmax(axis=1)]

    def update(self, dets: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        """Advance one frame. ``frame`` is unused by ByteTrack (DeepSORT needs it for features)."""
        self.frame_idx += 1
        out = self._bt.update_with_detections(self._to_sv(dets))
        tracks: list[Track] = []
        has_ids = out.tracker_id is not None and len(out) > 0
        ids = np.asarray(out.tracker_id, int) if has_ids else np.zeros(0, int)
        rows = self._source_rows(out, dets) if has_ids else []
        for row, tid in zip(rows, ids, strict=True):
            tid = int(tid)
            if tid < 0:  # unmatched rows, if this supervision version returns them
                continue
            det = dets[row]
            st = self._states.get(tid)
            if st is None:
                st = _TrackState(self.frame_idx, self.frame_idx, deque(maxlen=self.history_len))
                self._states[tid] = st
            st.last_frame = self.frame_idx
            st.history.append(det.bbox_xyxy.copy())
            tracks.append(
                Track(
                    track_id=tid,
                    det=det,
                    age=self.frame_idx - st.first_frame,
                    history=deque(st.history, maxlen=self.history_len),
                )
            )
        self._prune()
        return tracks
