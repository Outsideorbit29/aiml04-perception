"""F8.4: assemble the captioned demo video from the material of scripts/record_demo.sh.

    python scripts/make_demo_video.py [--config configs/demo_video.yaml]   # laptop, CPU, ~1 min

Segments (configs/demo_video.yaml): text cards, images, screen-capture frame sequences (played at
their captured timing, scaled to the segment length) and terminal cards (log or JSON files revealed
line by line). Each frame gets a caption band. Numbers from the run are read from the capture files.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

BG = (17, 20, 26)
PANEL = (28, 32, 40)
FG = (236, 238, 242)
DIM = (160, 168, 180)
ACCENT = (88, 166, 255)
TERM_BG = (22, 22, 22)
TERM_FG = (212, 212, 212)
TERM_HOT = (245, 169, 127)
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        names = ["consola.ttf", "DejaVuSansMono.ttf"]
    elif bold:
        names = ["segoeuib.ttf", "DejaVuSans-Bold.ttf"]
    else:
        names = ["segoeui.ttf", "DejaVuSans.ttf"]
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return ImageFont.truetype(str(d / n), size)
    return ImageFont.load_default()


# ---------------------------------------------------------------- run facts from the capture
def run_facts(cap: Path) -> dict[str, str]:
    facts: dict[str, str] = {}
    stats_path = cap / "consumer_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text())
        facts.update(
            processed=f"{s['processed']:,}",
            alerts=f"{s['alerts']:,}",
            e2e_p50=f"{s['e2e_ms']['p50']:.0f}",
            e2e_p95=f"{s['e2e_ms']['p95']:.0f}",
            http_p50=f"{s['http_ms']['p50']:.1f}",
            http_p95=f"{s['http_ms']['p95']:.1f}",
            capacity=f"{s['capacity_tps']:.0f}",
        )
    score_path = cap / "score.json"
    if score_path.exists():
        try:
            r = json.loads(score_path.read_text())
        except json.JSONDecodeError:
            r = {}
        if "score" in r:
            gnn = r.get("gnn_risk")
            gnn_txt = f"{gnn:.4f}" if isinstance(gnn, (int, float)) else "n/a (unseen client)"
            verdict = "alert" if r.get("alert") else "no alert"
            facts["score_caption"] = (
                f"One scoring call: blended score {r['score']:.4f} "
                f"(XGBoost {r.get('p_xgb', 0):.4f}, "
                f"client GNN risk {gnn_txt}) vs threshold {r.get('threshold', 0):.3f} → {verdict}; "
                f"{r.get('total_ms', 0):.1f} ms inside the scorer (Feast read + model)."
            )
        elif "http_status" in r:
            facts["score_caption"] = (
                f"HTTP {r['http_status']}: the scorer refuses when Redis holds another "
                "transaction's features for this client (stale-feature guard)."
            )
    facts.setdefault(
        "score_caption", "A /score call returns score, XGBoost part, GNN risk, alert flag, timings."
    )
    mon = cap / "monitor.log"
    amt = []
    fired = 0
    if mon.exists():
        for line in mon.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"TransactionAmt ([0-9.]+)", line)
            if m and "too small" not in line:
                amt.append(float(m.group(1)))
            if "[dry-run]" in line:
                fired += 1
    if amt:
        facts["drift_caption"] = (
            f"TransactionAmt PSI goes from {amt[0]:.2f} to {max(amt):.2f} once the drifted "
            f"amounts fill the window (threshold 0.2) → retrain decision, printed as a dry run "
            f"({fired}×)."
        )
    else:
        facts["drift_caption"] = "PSI per feature and on the score; retrain decisions are dry runs."
    return facts


def fill(text: str, run: dict[str, str]) -> str:
    return re.sub(r"\{run\.(\w+)\}", lambda m: run.get(m.group(1), "?"), text)


# ---------------------------------------------------------------- drawing
class Canvas:
    def __init__(self, w: int, h: int, cap_h: int):
        self.w, self.h, self.cap_h = w, h, cap_h
        self.area = (w, h - cap_h)
        self.f_cap = font(25)

    def caption(self, img: Image.Image, text: str) -> Image.Image:
        d = ImageDraw.Draw(img)
        d.rectangle([0, self.h - self.cap_h, self.w, self.h], fill=BG)
        d.line([0, self.h - self.cap_h, self.w, self.h - self.cap_h], fill=(60, 66, 78), width=2)
        lines = textwrap.wrap(text, width=92)[:3]
        lh = 31
        y = self.h - self.cap_h + (self.cap_h - lh * len(lines)) // 2
        for ln in lines:
            tw = d.textlength(ln, font=self.f_cap)
            d.text(((self.w - tw) / 2, y), ln, font=self.f_cap, fill=FG)
            y += lh
        return img

    def fit(self, src: Image.Image, crop_top: int = 0) -> Image.Image:
        if crop_top:
            src = src.crop((0, crop_top, src.width, src.height))
        aw, ah = self.area
        s = min(aw / src.width, ah / src.height)
        im = src.convert("RGB").resize((int(src.width * s), int(src.height * s)), Image.LANCZOS)
        out = Image.new("RGB", (self.w, self.h), BG)
        out.paste(im, ((aw - im.width) // 2, (ah - im.height) // 2))
        return out

    def card(self, title: str, lines: list[str]) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        aw, ah = self.area
        d.rounded_rectangle([40, 30, aw - 40, ah - 20], radius=18, fill=PANEL)
        d.text((80, 70), title, font=font(40, bold=True), fill=ACCENT)
        y = 150
        f = font(27)
        for ln in lines:
            for part in textwrap.wrap(ln, width=78) or [""]:
                d.text((80, y), part, font=f, fill=FG if part else DIM)
                y += 40
        return img

    def terminal(self, command: str, lines: list[str], shown: int) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), TERM_BG)
        d = ImageDraw.Draw(img)
        f = font(19, mono=True)
        d.text((30, 24), "$ " + command, font=font(20, mono=True), fill=ACCENT)
        y = 66
        for ln in lines[:shown]:
            for part in textwrap.wrap(ln, width=108, subsequent_indent="    ") or [""]:
                if y > self.area[1] - 26:
                    break
                hot = "dry-run" in ln or "retrain True" in ln or '"alert"' in ln
                d.text((30, y), part, font=f, fill=TERM_HOT if hot else TERM_FG)
                y += 25
        return img


# ---------------------------------------------------------------- segments
def load_manifest(cap: Path) -> list[dict[str, Any]]:
    path = cap / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def caption_at(seg: dict[str, Any], t: float, run: dict[str, str]) -> str:
    caps = seg.get("captions") or [{"at": 0, "text": seg.get("caption", "")}]
    text = caps[0]["text"]
    for c in caps:
        if t >= c["at"]:
            text = c["text"]
    return fill(text, run)


def term_lines(seg: dict[str, Any], path: Path) -> list[str]:
    if not path.exists():
        return [f"({path.name} not captured)"]
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if seg.get("grep"):
        lines = [x for x in lines if re.search(seg["grep"], x)]
    if seg.get("exclude"):
        lines = [x for x in lines if not re.search(seg["exclude"], x)]
    if seg.get("strip_prefix"):
        lines = [x.replace(seg["strip_prefix"], "", 1) for x in lines]
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", x) for x in lines if x.strip()]
    if path.suffix == ".json":
        try:
            lines = json.dumps(json.loads("\n".join(lines)), indent=2).splitlines()
        except json.JSONDecodeError:
            pass
    return lines[: seg.get("max_lines", 40)]


def render(cfg: dict[str, Any], repo: Path) -> tuple[Path, float]:
    w, h = cfg["size"]
    fps = cfg["fps"]
    cap = repo / cfg["capture_dir"]
    canvas = Canvas(w, h, cfg["caption_height"])
    run = run_facts(cap)
    manifest = load_manifest(cap)
    out = repo / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = 0.0
    cache: dict[str, Image.Image] = {}

    def emit(img: Image.Image) -> None:
        vw.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))

    for seg in cfg["segments"]:
        n = int(round(seg["seconds"] * fps))
        kind = seg["type"]
        if kind == "card":
            base = canvas.card(fill(seg["title"], run), [fill(x, run) for x in seg["lines"]])
            for i in range(n):
                emit(canvas.caption(base.copy(), caption_at(seg, i / fps, run)))
        elif kind == "image":
            p = Path(seg["path"].replace("{capture}", str(cap)))
            p = p if p.is_absolute() else repo / p
            base = canvas.fit(Image.open(p)) if p.exists() else canvas.card(f"missing {p.name}", [])
            for i in range(n):
                emit(canvas.caption(base.copy(), caption_at(seg, i / fps, run)))
        elif kind == "terminal":
            p = Path(seg["file"].replace("{capture}", str(cap)))
            lines = term_lines(seg, p)
            reveal = max(1, int(n * 0.6))
            for i in range(n):
                shown = min(len(lines), 1 + (len(lines) * i) // reveal)
                img = canvas.terminal(fill(seg["command"], run), lines, shown)
                emit(canvas.caption(img, caption_at(seg, i / fps, run)))
        elif kind == "frames":
            fr = [m for m in manifest if m["seg"] in seg["segs"]]
            if not fr:
                print(f"warning: no frames for {seg['segs']}")
                base = canvas.card("(not captured)", [", ".join(seg["segs"])])
                for i in range(n):
                    emit(canvas.caption(base.copy(), caption_at(seg, i / fps, run)))
                total += n / fps
                continue
            # Concatenate the sub-segments on one clock (drop the gaps between them).
            clock, last_t, last_seg, acc = [], None, None, 0.0
            for m in fr:
                if last_seg is not None and m["seg"] != last_seg:
                    acc += 0.5
                elif last_t is not None:
                    acc += m["t"] - last_t
                clock.append(acc)
                last_t, last_seg = m["t"], m["seg"]
            span = max(clock[-1] + 0.5, 1e-3)
            speed = span / seg["seconds"]
            crop = seg.get("crop_top", 0)
            j = 0
            for i in range(n):
                src_t = i / fps * speed
                while j + 1 < len(clock) and clock[j + 1] <= src_t:
                    j += 1
                f = fr[j]["file"]
                if f not in cache:
                    cache.clear()
                    cache[f] = canvas.fit(Image.open(cap / f), crop)
                emit(canvas.caption(cache[f].copy(), caption_at(seg, i / fps, run)))
            print(
                f"{'+'.join(seg['segs'])}: {len(fr)} frames, {span:.0f}s captured -> "
                f"{seg['seconds']}s (x{speed:.1f})"
            )
        else:
            raise ValueError(f"unknown segment type {kind}")
        total += n / fps
    vw.release()
    return out, total


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/demo_video.yaml")
    args = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / args.config).read_text(encoding="utf-8"))
    out, total = render(cfg, repo)
    print(f"wrote {out} ({total:.0f} s, {int(total // 60)}:{int(total % 60):02d})")


if __name__ == "__main__":
    main()
