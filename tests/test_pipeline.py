"""perception.pipeline (M4.6): wiring, depth skipping, JSONL, drawing and sources.

No torch / onnxruntime / supervision / model weights: the real Detector, DepthEstimator,
ClassicalLaneDetector and DistanceFusion run with fake backends; only the Tracker (supervision)
is replaced by a stub that gives each detection a fixed id.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.depth import DepthEstimator, load_camera
from perception.detection import Detector
from perception.detection.processing import RawDetections
from perception.fusion import DistanceFusion
from perception.lanes import build_lane_detector
from perception.pipeline import (
    Pipeline,
    draw_frame,
    dumps_frame,
    frame_to_dict,
    load_pipeline_config,
    open_source,
    run,
    snapshot_indices,
    stage_configs,
    summarize,
)
from perception.types import Detection, LaneResult, ObjectState, PerceptionFrame, Polyline, Track

FIX = Path(__file__).parent / "fixtures"
CFG = load_pipeline_config("configs/pipeline.yaml")
H, W = 450, 800  # CARLA camera (configs/camera.yaml carla: horizon at row 225)


class FakeDetBackend:
    """Fixed raw detections: a car whose box bottom is at row 300 (8.5 m with the CARLA camera)
    in the image centre, and a traffic light (no ground contact, never an ObjectState)."""

    names = None

    def __init__(self, scale: float = 1.0) -> None:
        s = scale
        self.raw = RawDetections(
            np.array([[350 * s, 240 * s, 450 * s, 300 * s], [100 * s, 20 * s, 120 * s, 60 * s]],
                     np.float32),
            np.array([0.9, 0.8], np.float32),
            np.array([0, 7]),
        )

    def __call__(self, frame: np.ndarray) -> RawDetections:
        return self.raw


class StubTracker:
    """Detection i -> track id i + 1 on every frame (ByteTrack needs supervision)."""

    cfg = {"frame_rate": 20}

    def __init__(self) -> None:
        self.n = 0

    def update(self, dets: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        self.n += 1
        return [Track(i + 1, d, age=self.n - 1) for i, d in enumerate(dets)]

    def reset(self) -> None:
        self.n = 0


class CountingDepth:
    """Relative inverse depth growing towards the image bottom; counts model calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        self.calls += 1
        _, _, h, w = blob.shape
        return np.tile(np.linspace(0.1, 1.0, h, dtype=np.float32)[:, None], (1, w))[None]


class StraightLanes:
    """Lane stub: vertical boundaries at x = 200 and x = 600, no mask."""

    name = "stub"
    last_debug = {"y_top": 250.0, "y_bottom": 449.0}

    def detect(self, frame: np.ndarray) -> LaneResult:
        return LaneResult(
            Polyline(np.array([0.0, 0.0, 200.0])), Polyline(np.array([0.0, 0.0, 600.0])), None, 0.9
        )


def make_pipeline(
    every_n: int = 1, lanes=None, camera: str = "carla", scale: float = 1.0, fps: float = 20.0
) -> tuple[Pipeline, CountingDepth, dict]:
    sc = stage_configs(CFG, camera_name=camera, lanes_method="classical", fps=fps)
    sc["depth"]["estimator"]["every_n_frames"] = every_n
    depth_backend = CountingDepth()
    pipe = Pipeline(
        detector=Detector(sc["detector"], backend=FakeDetBackend(scale)),
        tracker=StubTracker(),
        lanes=lanes if lanes is not None else build_lane_detector(sc["lanes"]),
        depth=DepthEstimator(sc["depth"], backend=depth_backend),
        fusion=DistanceFusion(sc["depth"], load_camera(CFG["camera_config"], camera)),
        fps=fps,
    )
    return pipe, depth_backend, sc


def frame(h: int = H, w: int = W) -> np.ndarray:
    return np.full((h, w, 3), 90, np.uint8)


def test_import_is_light():
    code = (
        "import sys, perception.pipeline; "
        "bad = [m for m in ('torch', 'ultralytics', 'onnxruntime', 'supervision', 'wandb') "
        "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    )
    root = Path(__file__).resolve().parents[1]
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert res.returncode == 0, res.stdout + res.stderr


def test_stage_configs_overrides():
    sc = stage_configs(CFG, backend="torch", camera_name="kitti", fps=10.0)
    assert sc["detector"]["backend"] == "torch"
    assert sc["depth"]["estimator"]["backend"] == "torch"
    assert sc["depth"]["models"]["midas_small"]["input_hw"] == [256, 832]
    assert sc["depth"]["models"]["midas_small"]["mean"]  # rest of the model section kept
    assert sc["lanes"]["method"] == "yolop"  # pipeline default
    assert sc["lanes"]["fitting"]["roi"]  # merged over configs/lanes.yaml, not replaced
    assert sc["tracker"]["frame_rate"] == 10
    assert sc["camera_name"] == "kitti"
    plain = stage_configs(CFG, lanes_method="classical")
    assert plain["lanes"]["method"] == "classical"
    assert plain["depth"]["models"]["midas_small"]["input_hw"] == [256, 448]  # carla: 16:9
    assert plain["detector"]["backend"] == "torch"  # module config default, untouched
    assert plain["depth"]["estimator"]["backend"] == "onnx"


def test_process_wires_perception_frame():
    pipe, _, _ = make_pipeline(lanes=StraightLanes())
    out = [pipe.process(frame()) for _ in range(4)]
    pf = out[-1]
    assert isinstance(pf, PerceptionFrame)
    assert pf.frame_id == 3 and pf.stamp == pytest.approx(3 / 20)
    assert [o.track.det.cls_name for o in pf.objects] == ["car"]  # light filtered by fusion
    car = pf.objects[0]
    assert car.distance_m == pytest.approx(8.53, rel=0.05)  # 1.6 * 400 / (300 - 225)
    assert car.in_ego_lane
    assert pf.lead_object() is car
    for k in ("detect_ms", "track_ms", "lanes_ms", "depth_ms", "fusion_ms", "total_ms"):
        assert pf.timings_ms[k] >= 0
    assert pf.timings_ms["total_ms"] >= pf.timings_ms["detect_ms"]
    assert [t.track_id for t in pipe.last_tracks] == [1, 2]
    assert pipe.last_stats["tracks"] == 2 and pipe.last_stats["objects"] == 1


def test_depth_every_n_frames_skips():
    pipe, depth, _ = make_pipeline(every_n=2)
    ran = []
    for _ in range(5):
        pipe.process(frame())
        ran.append(pipe.last_stats["depth_ran"])
    assert ran == [True, False, True, False, True]
    assert depth.calls == 3
    pipe.reset()
    pipe.process(frame())
    assert pipe.last_stats["depth_ran"] and pipe._frame_idx == 1


def test_no_depth_stage():
    pipe, _, _ = make_pipeline()
    pipe.depth = None
    pf = pipe.process(frame())
    assert len(pf.objects) == 1 and not pipe.last_stats["depth_ran"]


def test_jsonl_serialisation_is_strict_json():
    det = Detection(np.array([1, 2, 3, 4], np.float32), np.float32(0.7), 0, "car")
    obj = ObjectState(Track(5, det, age=3), np.float64(12.5), float("nan"), 1.0, None, True)
    lanes = LaneResult(Polyline(np.array([0.0, 0.1, 5.0])), None, np.ones((4, 4), bool), 0.4)
    pf = PerceptionFrame(1.5, 30, [obj], lanes, {"total_ms": np.float64(12.0)})
    line = dumps_frame(pf, {"tracks": 2, "depth_ran": True})
    assert "\n" not in line
    d = json.loads(line)
    assert d["frame_id"] == 30 and d["stamp"] == 1.5
    o = d["objects"][0]
    assert o["track_id"] == 5 and o["distance_std"] is None and o["ttc_s"] is None
    assert o["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0] and o["in_ego_lane"] is True
    assert d["lanes"] == {"left": [0.0, 0.1, 5.0], "right": None, "confidence": 0.4,
                          "has_ego_mask": True}
    assert d["lead_track_id"] == 5
    assert d["stats"] == {"tracks": 2, "depth_ran": True}
    assert frame_to_dict(pf)["timings_ms"] == {"total_ms": 12.0}


def test_draw_frame_on_synthetic_frame():
    pipe, _, sc = make_pipeline(lanes=StraightLanes())
    img = frame()
    for _ in range(4):
        pf = pipe.process(img)
    pf.objects[0].ttc_s = 1.5
    vis = draw_frame(img, pf, CFG["viz"], pipe.lane_debug(), pipe.last_tracks,
                     sc["lanes"]["viz"], info=["test"])
    assert vis.shape == img.shape and vis.dtype == np.uint8
    assert not np.array_equal(vis, img)
    assert np.array_equal(img, frame())  # input untouched
    # lead box outline (red, BGR) drawn around the car
    x1, y2 = 350, 300
    pad = CFG["viz"]["object_thickness"] + CFG["viz"]["lead_thickness"] // 2
    assert (vis[y2 - 30, x1 - pad] == np.array(CFG["viz"]["lead_color"])).all()
    # lane boundary at x = 200 drawn inside the measured rows only
    assert not np.array_equal(vis[400, 199:202], img[400, 199:202])
    assert np.array_equal(vis[150, 199:202], img[150, 199:202])
    # empty frame (no objects, no lanes) still draws
    empty = PerceptionFrame(0.0, 0, [], LaneResult(None, None, None, 0.0), {})
    assert draw_frame(img, empty, CFG["viz"]).shape == img.shape


def test_open_source_image_folder_and_video():
    s = CFG["source"]
    src = open_source(FIX / "images", 5.0, 10.0, s["image_suffixes"], max_frames=3)
    imgs = list(src.frames)
    assert src.fps == 5.0 and src.count == 3 and len(imgs) == 3
    assert imgs[0].ndim == 3 and imgs[0].dtype == np.uint8
    vid = open_source(FIX / "clip_kitti.mp4", 5.0, 7.0, s["image_suffixes"], max_frames=7)
    frames = list(vid.frames)
    assert vid.fps == pytest.approx(10.0) and vid.count == 7 and len(frames) == 7
    assert frames[0].shape[:2] == (144, 480)
    with pytest.raises(FileNotFoundError):
        open_source(FIX / "nope", 5.0, 5.0, s["image_suffixes"])


def test_run_on_fixture_clip_writes_outputs(tmp_path):
    s = CFG["source"]
    src = open_source(FIX / "clip_kitti.mp4", s["image_fps"], s["fallback_fps"],
                      s["image_suffixes"], max_frames=12)
    # KITTI clip is 480x144: detector boxes scaled from the 800x450 layout are fine for wiring
    pipe, _, sc = make_pipeline(camera="kitti", scale=0.32, fps=src.fps)
    out = tmp_path / "out.mp4"
    jl = tmp_path / "frames.jsonl"
    summary = run(pipe, src, CFG, out, jl, tmp_path / "snaps", sc["lanes"]["viz"])
    lines = jl.read_text().splitlines()
    assert len(lines) == 12
    assert [json.loads(x)["frame_id"] for x in lines] == list(range(12))
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok and first.shape[:2] == (144, 480) and n == 12
    snaps = sorted((tmp_path / "snaps").glob("*.jpg"))
    assert len(snaps) == CFG["output"]["snapshots"]
    assert summary["frames"] == 12
    assert summary["timed_frames"] == 12 - CFG["summary"]["warmup_frames"]
    assert summary["pipeline_fps"] > 0 and summary["depth_frames"] == 12


def test_summarize_and_snapshots():
    rec = {
        "timings_ms": {"detect_ms": 10.0, "total_ms": 20.0},
        "lanes": {"left": [0, 0, 1], "right": None, "has_ego_mask": True},
        "objects": [{"track_id": 1, "distance_m": 10.0, "ttc_s": 3.0, "in_ego_lane": True}],
        "lead_track_id": 1,
        "stats": {"tracks": 2, "objects": 1, "no_estimate": 1, "depth_ran": True, "aligned": 0},
    }
    slow = {**rec, "timings_ms": {"detect_ms": 1000.0, "total_ms": 2000.0}}
    s = summarize([slow, rec, rec, rec], warmup=1, wall_s=2.0)
    assert s["mean_ms"] == {"detect_ms": 10.0, "total_ms": 20.0}
    assert s["pipeline_fps"] == pytest.approx(50.0) and s["wall_fps"] == pytest.approx(2.0)
    assert s["frac_with_distance"] == pytest.approx(0.5)
    assert s["ttc_count"] == 4 and s["ttc_tracks"] == 1 and s["lead_frames"] == 4
    assert s["lanes_both_sides_frac"] == 0.0 and s["lanes_ego_mask_frac"] == 1.0
    assert snapshot_indices(50, 3) == {12, 24, 37}
    assert snapshot_indices(None, 3) == set() and snapshot_indices(10, 0) == set()
