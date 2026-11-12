"""Geometric distance, scale alignment and the DepthEstimator wrapper on synthetic data (M4.1-3).

Every test pins its own parameters instead of reading configs/*.yaml defaults, so retuning the
shipped configs cannot change what these tests check (only the SHIPPED test reads them).
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from perception.depth import (
    CameraModel,
    DepthEstimator,
    ScaleAligner,
    load_camera,
    load_depth_config,
    sample_box,
)

CAM = CameraModel(
    width=800,
    height=450,
    fx=400.0,
    fy=400.0,
    cx=400.0,
    cy=225.0,
    height_m=1.5,
    pitch_rad=0.0,
    sigma_v_px=1.0,
    sigma_pitch_rad=0.0,
)
ALIGN = {
    "min_points_affine": 3,
    "allow_scale_only": True,
    "huber_k": 1.5,
    "reject_k": 3.0,
    "irls_iters": 5,
    "sigma_floor": 1e-4,
    "min_inv_depth": 1e-3,
}


# ------------------------------------------------------------------ geometric distance


def test_ground_distance_matches_pinhole_formula():
    # Z = f h / (v - v_horizon): 400 * 1.5 / (255 - 225) = 20 m
    z, std = CAM.ground_distance(255.0)
    assert z == pytest.approx(20.0)
    # std = Z^2 / h * sigma_v / f = 400 / 1.5 / 400
    assert std == pytest.approx(20.0**2 / 1.5 * 1.0 / 400.0)


def test_ground_distance_of_projected_ground_point():
    """Project a 3D ground point with a pitched camera, then recover its distance."""
    pitch = math.radians(3.0)
    cam = CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.6, pitch)
    for z_true in (5.0, 12.0, 35.0):
        # ground point (forward z, down h) in a levelled frame -> camera frame tilted down
        y_lvl, z_lvl = 1.6, z_true
        y_c = math.cos(pitch) * y_lvl - math.sin(pitch) * z_lvl
        z_c = math.sin(pitch) * y_lvl + math.cos(pitch) * z_lvl
        v = cam.cy + cam.fy * y_c / z_c
        z, _ = cam.ground_distance(v)
        assert z == pytest.approx(z_true, rel=1e-9)
        assert cam.row_of_distance(z_true) == pytest.approx(v)
    assert cam.horizon_row == pytest.approx(225.0 - 400.0 * math.tan(pitch))


def test_rows_at_or_above_horizon_have_no_distance():
    assert CAM.ground_distance(225.0) is None
    assert CAM.ground_distance(100.0) is None
    # min angle: 1 deg below the horizontal is ~7 px at f=400
    assert CAM.ground_distance(228.0, min_angle_rad=math.radians(1.0)) is None
    assert CAM.ground_distance(240.0, min_angle_rad=math.radians(1.0)) is not None


def test_intrinsics_from_projection_and_rescale():
    P = np.array([[700.0, 0, 600.0, 45.0], [0, 710.0, 170.0, 0.0], [0, 0, 1.0, 0.0]])
    cam = CAM.with_intrinsics(P, (370, 1220))
    assert (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height) == (
        700.0,
        710.0,
        600.0,
        170.0,
        1220,
        370,
    )
    half = CAM.scaled_to((225, 400))
    assert (half.fx, half.cy) == (200.0, 112.5)
    # same scene point at half resolution -> same distance
    assert half.ground_distance(127.5)[0] == pytest.approx(CAM.ground_distance(255.0)[0])


def test_load_camera_requires_keys(tmp_path):
    p = tmp_path / "cam.yaml"
    p.write_text("x:\n  width: 10\n")
    with pytest.raises(KeyError):
        load_camera(p, "x")
    with pytest.raises(KeyError):
        load_camera(p, "missing")


# ------------------------------------------------------------------ scale alignment


def test_alignment_recovers_known_scale_and_shift():
    z = np.array([4.0, 7.0, 11.0, 18.0, 30.0])
    s_true, t_true = 0.002, -0.01  # 1/z = s m + t  ->  m = (1/z - t) / s
    m = (1.0 / z - t_true) / s_true
    a = ScaleAligner(ALIGN).fit(m, z)
    assert a is not None and a.mode == "affine"
    assert a.s == pytest.approx(s_true, rel=1e-6)
    assert a.t == pytest.approx(t_true, abs=1e-9)
    np.testing.assert_allclose(a.to_depth(m), z, rtol=1e-6)
    assert a.rms == pytest.approx(0.0, abs=1e-9)
    assert a.depth_std(10.0) == pytest.approx(100.0 * ALIGN["sigma_floor"])


def test_alignment_rejects_outlier_anchor():
    z = np.array([5.0, 8.0, 12.0, 16.0, 22.0, 28.0])
    m = 500.0 / z + 3.0  # s = 1/500 after removing the shift
    z_bad = z.copy()
    z_bad[2] = 40.0  # one anchor on a hill: geometric distance far off
    a = ScaleAligner(ALIGN).fit(m, z_bad)
    assert a is not None and a.n_inliers == 5
    np.testing.assert_allclose(a.to_depth(m)[[0, 1, 3, 4, 5]], z[[0, 1, 3, 4, 5]], rtol=1e-6)
    assert a.to_depth(m[2]) == pytest.approx(12.0, rel=1e-6)  # outlier object fixed by MiDaS


def test_alignment_fallbacks():
    al = ScaleAligner(ALIGN)
    assert al.fit(np.array([]), np.array([])) is None
    a = al.fit(np.array([50.0, 25.0]), np.array([10.0, 20.0]))  # 2 anchors < 3: scale only
    assert a.mode == "scale" and a.t == 0.0
    assert a.s == pytest.approx(1.0 / 500.0)
    assert ScaleAligner({**ALIGN, "allow_scale_only": False}).fit([50.0], [10.0]) is None
    # negative slope (MiDaS ordering inverted) is rejected, not used
    assert (
        ScaleAligner({**ALIGN, "allow_scale_only": False}).fit(
            [10.0, 20.0, 30.0], [5.0, 10.0, 20.0]
        )
        is None
    )
    # far values beyond 1/min_inv_depth give NaN instead of a huge distance
    a = al.fit([100.0, 50.0, 25.0], [5.0, 10.0, 20.0])
    assert np.isnan(a.to_depth(0.01))


def test_alignment_rejects_bad_config():
    with pytest.raises(KeyError):
        ScaleAligner({"min_points_affine": 3})
    with pytest.raises(ValueError):
        ScaleAligner({**ALIGN, "min_points_affine": 1})


# ------------------------------------------------------------------ estimator + sampling

MODEL = {
    "input_hw": [64, 96],
    "multiple_of": 32,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "resize_interp": "linear",
}
EST_CFG = {"estimator": {"model": "fake", "backend": "torch"}, "models": {"fake": MODEL}}


class FakeBackend:
    """Inverse depth that grows towards the image bottom (closer road), shape (1, h, w)."""

    def __init__(self):
        self.seen = None

    def __call__(self, blob):
        self.seen = blob
        _, _, h, w = blob.shape
        return np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))[None]


def test_estimator_shapes_and_normalisation():
    fake = FakeBackend()
    est = DepthEstimator(EST_CFG, backend=fake)
    frame = np.full((120, 200, 3), 255, np.uint8)
    out = est.estimate(frame)
    assert out.shape == (120, 200) and out.dtype == np.float32
    assert fake.seen.shape == (1, 3, 64, 96) and fake.seen.dtype == np.float32
    np.testing.assert_allclose(fake.seen, 1.0)  # (1 - 0.5) / 0.5
    assert out[-1, 0] > out[0, 0]
    assert set(est.last_timing_ms) == {"pre_ms", "model_ms", "post_ms", "total_ms"}


def test_estimator_every_n_frames_and_config_checks():
    cfg = {**EST_CFG, "estimator": {**EST_CFG["estimator"], "every_n_frames": 3}}
    est = DepthEstimator(cfg, backend=FakeBackend())
    frame = np.zeros((64, 96, 3), np.uint8)
    runs = [est.maybe_estimate(frame) is not None for _ in range(6)]
    assert runs == [True, False, False, True, False, False]
    bad = {**EST_CFG, "models": {"fake": {**MODEL, "input_hw": [60, 96]}}}
    with pytest.raises(ValueError):
        DepthEstimator(bad, backend=FakeBackend())
    with pytest.raises(ValueError):
        DepthEstimator({**EST_CFG, "estimator": {"model": "fake", "backend": "nope"}})
    with pytest.raises(KeyError):
        DepthEstimator({**EST_CFG, "estimator": {"model": "other", "backend": "torch"}})


def test_sample_box_uses_lower_central_part():
    inv = np.zeros((100, 100), np.float32)
    inv[50:, :] = 2.0  # lower half of the image
    inv[60:80, 40:60] = 5.0  # the object
    scfg = {"rows_frac": [0.5, 1.0], "cols_frac": [0.25, 0.75], "percentile": 50}
    assert sample_box(inv, np.array([30, 40, 70, 80]), scfg) == pytest.approx(5.0)
    assert math.isnan(sample_box(inv, np.array([200, 200, 210, 210]), scfg))


def test_importing_depth_and_fusion_does_not_import_torch():
    code = (
        "import sys; import perception.depth, perception.fusion; "
        "assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_shipped_configs_load():
    cfg = load_depth_config("configs/depth.yaml")
    for name, m in cfg["models"].items():
        assert all(v % m["multiple_of"] == 0 for v in m["input_hw"]), name
    ScaleAligner(cfg["alignment"])
    for name in ("kitti", "carla", "bdd"):
        cam = load_camera("configs/camera.yaml", name)
        assert 0 < cam.horizon_row < cam.height
    carla = load_camera("configs/camera.yaml", "carla")
    assert carla.fx == pytest.approx(carla.width / 2 / math.tan(math.radians(45)))
