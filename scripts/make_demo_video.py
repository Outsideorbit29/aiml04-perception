"""M7.4: assemble the demo video (and a README GIF) from the CARLA scenario recordings.

    python scripts/make_demo_video.py --config configs/demo_video.yaml     # laptop, CPU only

Clips are placed on a 1280x720 canvas with a caption band; title, overlay-legend, results and
benchmark cards are drawn with Pillow. No re-timing: clips play at their recorded 20 fps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

BG = (24, 26, 30)
FG = (240, 240, 240)
ACCENT = (90, 180, 255)
DIM = (170, 175, 185)
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "DejaVuSans.ttf"]
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return ImageFont.truetype(str(d / n), size)
    return ImageFont.load_default()


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def text_card(size: tuple[int, int], lines: list[str]) -> np.ndarray:
    """First line large (accent), the rest normal, vertically centred."""
    w, h = size
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    fonts = [font(46, True)] + [font(28)] * (len(lines) - 1)
    heights = [f.getbbox("Ag")[3] + 22 for f in fonts]
    y = (h - sum(heights)) // 2
    for i, (line, f) in enumerate(zip(lines, fonts, strict=True)):
        tw = d.textlength(line, font=f)
        d.text(((w - tw) / 2, y), line, font=f, fill=ACCENT if i == 0 else FG)
        y += heights[i] + (16 if i == 0 else 0)
    return to_bgr(img)


def table_card(size: tuple[int, int], spec: dict[str, Any]) -> np.ndarray:
    w, h = size
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    ft, fh, fr, ff = font(40, True), font(26, True), font(26), font(22)
    d.text(
        ((w - d.textlength(spec["title"], font=ft)) / 2, 110), spec["title"], font=ft, fill=ACCENT
    )
    cols = [140, 520, 700, 880, 1060]
    y = 220
    for x, cell in zip(cols, spec["header"], strict=True):
        d.text((x, y), cell, font=fh, fill=DIM)
    d.line([(cols[0], y + 44), (1180, y + 44)], fill=DIM, width=2)
    for row in spec["rows"]:
        y += 70
        for j, (x, cell) in enumerate(zip(cols, row, strict=True)):
            bold = j in (3, 4)
            d.text((x, y), cell, font=fh if bold else fr, fill=ACCENT if bold else FG)
    d.text(
        ((w - d.textlength(spec["footer"], font=ff)) / 2, y + 120),
        spec["footer"],
        font=ff,
        fill=DIM,
    )
    return to_bgr(img)


def frame_with_caption(
    frame: np.ndarray, size: tuple[int, int], cap_h: int, caption: str
) -> np.ndarray:
    w, h = size
    vh = h - cap_h
    scale = min(w / frame.shape[1], vh / frame.shape[0])
    fw, fh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    canvas = np.full((h, w, 3), BG[::-1], np.uint8)
    x0 = (w - fw) // 2
    canvas[0:fh, x0 : x0 + fw] = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f = font(26)
    tw = d.textlength(caption, font=f)
    d.text(((w - tw) / 2, fh + (cap_h - 34) / 2), caption, font=f, fill=FG)
    return to_bgr(img)


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demo_video.yaml"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    size = tuple(cfg["size"])
    fps = float(cfg["fps"])
    card_n = int(cfg["card_seconds"] * fps)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0

    def hold(img: np.ndarray, frames: int) -> None:
        nonlocal n
        for _ in range(frames):
            vw.write(img)
        n += frames

    hold(text_card(size, cfg["title"]), card_n)
    hold(text_card(size, cfg["overlay_card"]), int(card_n * 1.6))
    gif_frames: list[np.ndarray] = []
    for clip in cfg["clips"]:
        hold(text_card(size, clip["card"]), int(card_n * 0.6))
        frames = read_frames(clip["file"])
        for f in frames:
            vw.write(frame_with_caption(f, size, int(cfg["caption_height"]), clip["caption"]))
        n += len(frames)
        hold(
            frame_with_caption(frames[-1], size, int(cfg["caption_height"]), clip["caption"]),
            int(fps),
        )
        if clip["name"] == cfg["gif_clip"]:
            a, b = (int(s * fps) for s in cfg["gif_seconds"])
            gif_frames = frames[a:b]
    hold(text_card(size, cfg["results_card"]), int(card_n * 1.4))
    hold(table_card(size, cfg["bench_card"]), int(card_n * 1.6))
    hold(text_card(size, cfg["end_card"]), card_n)
    vw.release()
    print(f"wrote {out}: {n} frames, {n / fps:.1f} s, {out.stat().st_size / 1e6:.1f} MB")

    if gif_frames and cfg.get("gif"):
        gw = int(cfg["gif_width"])
        imgs = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize(
                (gw, int(f.shape[0] * gw / f.shape[1]))
            )
            for f in gif_frames[::2]
        ]
        gp = Path(cfg["gif"])
        imgs[0].save(gp, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
        print(f"wrote {gp}: {len(imgs)} frames, {gp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
