"""Draw tracks on a frame (annotated video on Kaggle, debug image in ROS2)."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from perception.types import Track

GOLDEN = 0.618033988749895  # spreads hues of consecutive ids around the colour wheel


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, well separated BGR colour per track id."""
    r, g, b = colorsys.hsv_to_rgb((track_id * GOLDEN) % 1.0, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    trail: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Return a copy of ``frame`` with boxes, ``id class score`` labels and bottom-centre trails."""
    out = frame.copy()
    for t in tracks:
        color = id_color(t.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in t.det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{t.track_id} {t.det.cls_name} {t.det.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1, th + base)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(
            out, label, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1,
            cv2.LINE_AA,
        )
        if trail and len(t.history) > 1:
            pts = np.array([[(b[0] + b[2]) / 2, b[3]] for b in t.history], np.int32)
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color, max(1, thickness - 1))
    return out
