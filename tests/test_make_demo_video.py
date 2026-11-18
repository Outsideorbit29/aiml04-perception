"""M7.4 demo video builder: config consistency and a tiny end-to-end build on synthetic clips."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("PIL")  # Pillow comes with supervision (requirements-dev.txt)

from scripts.make_demo_video import main, table_card, text_card  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "demo_video.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def test_shipped_config_is_consistent():
    cfg = _cfg()
    names = [c["name"] for c in cfg["clips"]]
    assert len(names) == len(set(names)) == 4
    assert cfg["gif_clip"] in names
    a, b = cfg["gif_seconds"]
    assert 0 <= a < b
    bench = cfg["bench_card"]
    # table_card zips columns with strict=True, so a short row would crash the build
    assert all(len(r) == len(bench["header"]) for r in bench["rows"])
    for key in ("title", "overlay_card", "results_card", "end_card"):
        assert cfg[key] and all(isinstance(s, str) and s for s in cfg[key])


def test_shipped_clips_are_logged_runs():
    """Every clip the video uses is a run logged in docs/EXPERIMENTS.md (no unlogged footage)."""
    log = (ROOT / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8")
    for clip in _cfg()["clips"]:
        assert Path(clip["file"]).name in log, clip["file"]


def test_cards_have_canvas_shape():
    cfg = _cfg()
    size = (320, 180)
    for img in (text_card(size, cfg["title"]), table_card(size, cfg["bench_card"])):
        assert img.shape == (180, 320, 3) and img.dtype == np.uint8
        assert img.std() > 0  # something was drawn


def _write_clip(path: Path, n: int, fps: float) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    assert vw.isOpened()
    for i in range(n):
        f = np.full((90, 160, 3), (i * 8) % 255, np.uint8)
        vw.write(f)
    vw.release()


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_build_on_synthetic_clips(tmp_path, capsys):
    cfg = _cfg()
    fps, clip_n = 10.0, 12
    clips = []
    for name in ("a", "b"):
        p = tmp_path / f"{name}.mp4"
        _write_clip(p, clip_n, fps)
        clips.append({"name": name, "file": str(p), "card": ["Card", name], "caption": "cap"})
    cfg.update(
        out=str(tmp_path / "out" / "demo.mp4"),
        gif=str(tmp_path / "out" / "demo.gif"),
        gif_clip="b",
        gif_seconds=[0.0, 1.0],
        gif_width=80,
        size=[320, 180],
        fps=fps,
        card_seconds=0.5,
        caption_height=30,
        clips=clips,
    )
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    main(["--config", str(cfg_path)])

    card_n = int(0.5 * fps)
    per_clip = int(card_n * 0.6) + clip_n + int(fps)
    expected = (
        card_n + int(card_n * 1.6) + 2 * per_clip + int(card_n * 1.4) + int(card_n * 1.6) + card_n
    )
    out = tmp_path / "out" / "demo.mp4"
    assert f"{expected} frames" in capsys.readouterr().out
    assert _count_frames(out) == expected
    gif = tmp_path / "out" / "demo.gif"
    assert gif.is_file() and gif.stat().st_size > 0


def test_missing_clip_raises(tmp_path):
    cfg = _cfg()
    cfg.update(
        out=str(tmp_path / "demo.mp4"),
        gif=None,
        size=[320, 180],
        card_seconds=0.1,
        clips=[{"name": "x", "file": str(tmp_path / "nope.mp4"), "card": ["x"], "caption": "c"}],
    )
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        main(["--config", str(cfg_path)])
