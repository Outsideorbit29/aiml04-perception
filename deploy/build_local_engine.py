"""Build TensorRT engines for THIS GPU from the exported ONNX files (M6.2 on Kaggle, M6.4 laptop).

    # WSL2 Ubuntu 22.04 on the laptop, once (~5 min); needs: pip install tensorrt-cu12 cuda-python
    python3 deploy/build_local_engine.py --onnx models/*.onnx --fp16
    # Kaggle T4 (kaggle/export_bench): same script, engines next to the job's ONNX files
    python deploy/build_local_engine.py --onnx /kaggle/temp/m6_models/*.onnx --fp16

Writes ``<stem>.engine`` (plus ``<stem>.engine.json`` with the build settings and versions) next
to each ONNX file or into ``--out-dir``. These are the names the runtime backends look for:
``models/yolo11s-bdd.engine`` (configs/yolo_detector.yaml weights with the .engine suffix),
``models/midas_small.engine`` (configs/depth.yaml onnx path with .engine) and
``models/yolop-640-640.engine``. Settings match ``trtexec --fp16 --memPoolSize=workspace:1024M``.
Engines are hardware-specific and gitignored; rerun this script after a driver or TensorRT
upgrade. No PyTorch needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.trt_build import build_engine, engine_path_for  # noqa: E402


def expand(patterns: list[str]) -> list[Path]:
    """Expand globs (PowerShell / cmd do not) and drop duplicates, keeping order."""
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or [pat]
        for h in hits:
            p = Path(h)
            if p not in out:
                out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", nargs="+", required=True, help="ONNX files or globs")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: next to each ONNX")
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    prec.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--workspace-mib", type=int, default=1024, help="builder workspace limit")
    ap.add_argument("--force", action="store_true", help="rebuild even if the engine exists")
    ap.add_argument("--summary", type=Path, default=None, help="write all build summaries here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    files = expand(args.onnx)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print(f"ONNX files not found: {missing}", file=sys.stderr)
        return 2
    summaries = []
    for onnx in files:
        engine = engine_path_for(onnx, args.out_dir)
        if engine.exists() and not args.force:
            print(f"skip {engine} (exists; --force rebuilds)", flush=True)
            continue
        print(f"building {engine} from {onnx} ({'fp16' if args.fp16 else 'fp32'}, "
              f"workspace {args.workspace_mib} MiB)", flush=True)
        s = build_engine(onnx, engine, args.fp16, args.workspace_mib, args.verbose)
        print(json.dumps(s), flush=True)
        summaries.append(s)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
