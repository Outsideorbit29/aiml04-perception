"""Pure-python helpers for converting BDD100K detection labels to YOLO format.

No numpy/cv2/PIL here, so the helpers are unit-tested on CPU in CI with tiny fixtures.
Handles both BDD100K label layouts:
- 2018 release (``bdd100k_labels_images_{train,val}.json``): categories ``person``, ``bike``,
  ``motor``, plus ``lane`` / ``drivable area`` polygons that are skipped.
- det_20 (``det_{train,val}.json``): categories ``pedestrian``, ``bicycle``, ``motorcycle``,
  ``other vehicle``, ``other person``, ``trailer``.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

# Index order is fixed by ARCHITECTURE.md section 3.1. Do not reorder.
CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "person",
    "rider",
    "bicycle",
    "motorcycle",
    "traffic light",
    "traffic sign",
)

# BDD category name (any release) -> project class name. Anything else is dropped
# (train, trailer, other vehicle, other person, lane, drivable area, ...).
BDD_TO_CLASS: dict[str, str] = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "person": "person",
    "pedestrian": "person",
    "rider": "rider",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}

# COCO area thresholds (pixels^2, measured at the original resolution).
SMALL_AREA = 32**2
MEDIUM_AREA = 96**2


def class_id(category: str, mapping: dict[str, str] | None = None) -> int | None:
    """Return the project class index for a BDD category, or None if it is dropped."""
    name = (mapping or BDD_TO_CLASS).get(category.strip().lower())
    if name is None:
        return None
    return CLASSES.index(name)


def clip_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Clip an xyxy box to the image and make sure x1<=x2, y1<=y2."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a pixel xyxy box to normalised YOLO (cx, cy, w, h) in [0, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def area_bucket(area: float) -> str:
    """COCO size bucket for a box area in pixels^2."""
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def iter_boxes(frame: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield label dicts with a ``box2d`` from either the flat or the per-frame layout."""
    labels = frame.get("labels")
    if labels is None and frame.get("frames"):  # per-image JSON files of the 2018 release
        labels = frame["frames"][0].get("objects", [])
    for lab in labels or []:
        if lab.get("box2d"):
            yield lab


def frame_attributes(frame: dict[str, Any]) -> tuple[str, str]:
    """Return (timeofday, weather) with 'undefined' for missing values."""
    attrs = frame.get("attributes") or {}
    return attrs.get("timeofday") or "undefined", attrs.get("weather") or "undefined"


def frame_to_yolo(
    frame: dict[str, Any],
    width: int,
    height: int,
    min_size_px: float = 1.0,
    mapping: dict[str, str] | None = None,
) -> list[tuple[int, float, float, float, float]]:
    """Convert one BDD frame to YOLO rows ``(cls, cx, cy, w, h)``.

    Boxes of dropped categories, and boxes thinner than ``min_size_px`` after clipping,
    are skipped.
    """
    rows = []
    for lab in iter_boxes(frame):
        cid = class_id(lab.get("category", ""), mapping)
        if cid is None:
            continue
        b = lab["box2d"]
        x1, y1, x2, y2 = clip_box(b["x1"], b["y1"], b["x2"], b["y2"], width, height)
        if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
            continue
        rows.append((cid, *xyxy_to_yolo(x1, y1, x2, y2, width, height)))
    return rows


def format_yolo_rows(rows: Sequence[tuple[int, float, float, float, float]]) -> str:
    """Serialise YOLO rows to label-file text (one object per line, 6 decimals)."""
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows)


def allocate(counts: dict[Hashable, int], total: int, min_per_stratum: int = 1) -> dict:
    """Split ``total`` samples across strata proportionally (largest-remainder method).

    Every non-empty stratum gets at least ``min(count, min_per_stratum)`` (dropped if the
    minimums alone exceed ``total``); no stratum gets more than it has. If ``total`` >= the
    population, every stratum is taken in full. Ties are broken by the stratum key string.
    """
    population = sum(counts.values())
    if total >= population:
        return dict(counts)
    if total <= 0:
        return {k: 0 for k in counts}
    floor_of = {k: min(c, min_per_stratum) for k, c in counts.items()}
    if sum(floor_of.values()) > total:
        floor_of = {k: 0 for k in counts}
    quota = {k: total * c / population for k, c in counts.items()}
    alloc = {k: min(counts[k], max(floor_of[k], int(quota[k]))) for k in counts}
    diff = total - sum(alloc.values())
    while diff > 0:  # hand out the leftover units to the largest remainders
        order = sorted(
            (k for k in counts if alloc[k] < counts[k]),
            key=lambda k: (-(quota[k] - alloc[k]), str(k)),
        )
        for k in order[:diff]:
            alloc[k] += 1
        diff = total - sum(alloc.values())
    while diff < 0:  # minimums pushed us over: take back from the most over-allocated
        order = sorted(
            (k for k in counts if alloc[k] > floor_of[k]),
            key=lambda k: (quota[k] - alloc[k], str(k)),
        )
        for k in order[:-diff]:
            alloc[k] -= 1
        diff = total - sum(alloc.values())
    return alloc


def stratified_sample(
    items: Sequence[Any],
    key: Callable[[Any], Hashable],
    total: int,
    seed: int = 42,
    min_per_stratum: int = 1,
) -> list[Any]:
    """Draw ``total`` items stratified by ``key`` with a fixed seed.

    Deterministic for a given input order and seed. The result keeps the input order.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        groups[key(item)].append(i)
    alloc = allocate({k: len(v) for k, v in groups.items()}, total, min_per_stratum)
    rng = random.Random(seed)
    chosen: list[int] = []
    for k in sorted(groups, key=str):
        chosen.extend(rng.sample(groups[k], alloc[k]))
    return [items[i] for i in sorted(chosen)]


def stratum_counts(frames: Iterable[dict[str, Any]]) -> Counter:
    """Count frames per (timeofday, weather)."""
    return Counter(frame_attributes(f) for f in frames)
