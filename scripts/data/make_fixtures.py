"""Make the tiny test fixtures (runs on Kaggle after ``prepare_bdd``; CPU only).

- 5 BDD100K val images, downscaled to ``--width`` px wide JPEG, with their YOLO labels
  (normalised coordinates, so the labels are unchanged by the resize). Chosen to cover
  day/night/rain and the rare classes rider/motorcycle.
- A short clip (``--clip-frames`` consecutive frames at 10 fps) from a KITTI tracking sequence,
  written as MP4 (mp4v).

    python -m scripts.data.make_fixtures --data /kaggle/working/bdd_yolo \
        --kitti-seq /kaggle/input/.../training/image_02/0000 --out /kaggle/working/fixtures
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.data.bdd_yolo import CLASSES

WANTED = [
    ("daytime", "clear", "rider"),
    ("night", "clear", "motorcycle"),
    ("daytime", "rainy", "person"),
    ("night", "rainy", "car"),
    ("dawn/dusk", None, "bus"),
]


def pick_frames(frames: list[dict], seed_names: int = 0) -> list[dict]:
    picks, used = [], set()
    for tod, weather, cls in WANTED:
        cid = CLASSES.index(cls)
        for fr in frames:
            if fr["name"] in used or fr["timeofday"] != tod:
                continue
            if weather and fr["weather"] != weather:
                continue
            ids = [b[0] for b in fr["boxes"]]
            if cid in ids and 3 <= len(ids) <= 25:
                picks.append(fr)
                used.add(fr["name"])
                break
    return picks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepare_bdd output folder")
    ap.add_argument("--kitti-seq", type=Path, help="folder of consecutive KITTI frames")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--clip-width", type=int, default=480)
    ap.add_argument("--clip-frames", type=int, default=30)
    ap.add_argument("--clip-fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    import cv2

    img_out, lbl_out = args.out / "images", args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.data / "meta" / "frames_val.jsonl.gz", "rt", encoding="utf-8") as fh:
        frames = [json.loads(line) for line in fh]
    manifest = {"images": [], "clip": None}
    for i, fr in enumerate(pick_frames(frames)):
        src = args.data / "images" / "val" / fr["name"]
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        size = (args.width, round(h * args.width / w))
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        stem = f"bdd_{i}_{Path(fr['name']).stem}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        label = (args.data / "labels" / "val" / (Path(fr["name"]).stem + ".txt")).read_text()
        (lbl_out / f"{stem}.txt").write_text(label)
        manifest["images"].append(
            {
                "file": f"images/{stem}.jpg",
                "source_name": fr["name"],
                "timeofday": fr["timeofday"],
                "weather": fr["weather"],
                "n_boxes": len(label.splitlines()),
                "size": [small.shape[1], small.shape[0]],
            }
        )
    if args.kitti_seq:
        files = sorted(args.kitti_seq.glob("*.png"))[: args.clip_frames]
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]
        size = (args.clip_width, round(h * args.clip_width / w) // 2 * 2)
        path = args.out / "clip_kitti.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.clip_fps, size)
        for f in files:
            vw.write(cv2.resize(cv2.imread(str(f)), size, interpolation=cv2.INTER_AREA))
        vw.release()
        cap = cv2.VideoCapture(str(path))
        n_read = 0
        while cap.read()[0]:
            n_read += 1
        manifest["clip"] = {
            "file": "clip_kitti.mp4",
            "source": str(args.kitti_seq),
            "frames": len(files),
            "frames_read_back": n_read,
            "fps": args.clip_fps,
            "size": list(size),
        }
    (args.out / "fixtures_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
