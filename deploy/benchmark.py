"""M6.3: PyTorch vs ONNX Runtime vs TensorRT on the Kaggle T4 (the official before/after table).

    python deploy/benchmark.py --config configs/bench.yaml [--smoke]

Runs after deploy/export.py (ONNX files) and deploy/build_local_engine.py (T4 FP16 engines) in
kaggle/export_bench; never on the laptop. For every model (YOLO11s, MiDaS-small, YOLOP) and
backend (torch FP32, onnx = ONNX Runtime CUDA EP FP32, tensorrt = TensorRT FP16) one worker
process (``--worker``) is started, so peak VRAM (NVML, CUDA context included) and library state
are per backend. Each worker:
1. builds the live module with that backend: ``Detector`` / ``DepthEstimator`` /
   ``YolopLaneDetector``, all with the same runtime pre/post-processing;
2. times ``bench.warmup`` + ``bench.iters`` iterations of (a) the model alone on a fixed
   preprocessed tensor, host in / host out (copies included), and (b) the whole module call on a
   1280x720 BDD frame (preprocess + model + post-process);
3. parity on the fixture frames: max / mean abs diff of every output vs the PyTorch output;
4. accuracy on a BDD val subset: YOLO mAP@0.5 and mAP@0.5:0.95 (val settings), MiDaS AbsRel and
   YOLOP lane-mask IoU against the PyTorch output (the reference worker saves those).
The orchestrator writes ``out_dir``/results.json + benchmarks.md (copied into docs/benchmarks.md)
and logs them to W&B group ``bench``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from deploy.common import (  # noqa: E402
    BACKENDS,
    MODELS,
    VramSampler,
    blob_fn,
    engine_path,
    fixture_frames,
    git_sha,
    load_bench_config,
    local_val_pairs,
    nvidia_smi,
    onnx_path,
    package_versions,
    runtime_configs,
    stage_val_subset,
)
from deploy.metrics import (  # noqa: E402
    MapAccumulator,
    abs_rel,
    latency_stats,
    merge_parity,
    parity,
)

MODEL_LABELS = {"yolo": "YOLO11s (BDD)", "midas_small": "MiDaS-small", "yolop": "YOLOP"}
BACKEND_LABELS = {
    "torch": "PyTorch FP32",
    "onnx": "ONNX Runtime CUDA FP32",
    "tensorrt": "TensorRT FP16",
}
INPUT_LABELS = {"yolo": "640x640", "midas_small": "256x448", "yolop": "640x640"}


# ---------------------------------------------------------------- report (pure, unit-tested)


def _f(v: Any, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def accuracy_cells(model: str, r: dict[str, Any], ref: dict[str, Any] | None) -> tuple[str, str]:
    """(accuracy, delta vs PyTorch) cells of one result row."""
    acc = r.get("accuracy") or {}
    if model == "yolo" and "map50" in acc:
        cell = f"mAP50 {acc['map50']:.4f} / 50-95 {acc['map50_95']:.4f}"
        racc = (ref or {}).get("accuracy") or {}
        if r.get("backend") == "torch" or "map50" not in racc:
            return cell, "ref" if r.get("backend") == "torch" else "–"
        return cell, (f"{acc['map50'] - racc['map50']:+.4f} / "
                      f"{acc['map50_95'] - racc['map50_95']:+.4f}")
    if model == "midas_small" and "abs_rel_vs_ref" in acc:
        if acc.get("is_reference"):
            return "AbsRel vs ref 0", "ref"
        return f"AbsRel vs ref {acc['abs_rel_vs_ref']:.4f}", f"{acc['abs_rel_vs_ref']:+.4f}"
    if model == "yolop" and "lane_iou_vs_ref" in acc:
        if acc.get("is_reference"):
            return "lane IoU vs ref 1", "ref"
        return (f"lane IoU vs ref {acc['lane_iou_vs_ref']:.4f}",
                f"{acc['lane_iou_vs_ref'] - 1:+.4f}")
    return "–", "–"


def parity_cell(r: dict[str, Any]) -> str:
    par = r.get("parity") or {}
    if r.get("write_ref"):
        return "ref"
    if not par:
        return "–"
    return ", ".join(f"{k} {v['max_abs']:.3g}" for k, v in par.items())


def markdown_report(results: Sequence[dict[str, Any]], meta: dict[str, Any]) -> str:
    """The benchmark table (one row per model x backend) plus environment and notes."""
    by = {(r["model"], r["backend"]): r for r in results}
    lines = [
        "| Model (input) | Backend | Model p50 ms | Model p95 ms | Model FPS | Speed-up vs "
        "PyTorch (p50) | Module p50 ms | Module FPS | Peak VRAM MB | Accuracy | Δ vs PyTorch "
        "| Parity max abs diff (fixtures) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        ref = by.get((model, meta.get("reference", {}).get(model, "torch")))
        base = by.get((model, "torch"))
        base_p50 = ((base or {}).get("latency_model") or {}).get("p50_ms")
        for backend in BACKENDS:
            r = by.get((model, backend))
            if r is None:
                continue
            name = f"{MODEL_LABELS[model]} ({INPUT_LABELS[model]})"
            if not r.get("ok"):
                err = str(r.get("error", "failed")).replace("|", "/")[:120]
                lines.append(f"| {name} | {BACKEND_LABELS[backend]} | failed: {err} |"
                             + " |" * 9)
                continue
            lm, le = r.get("latency_model") or {}, r.get("latency_e2e") or {}
            p50 = lm.get("p50_ms")
            speed = f"{base_p50 / p50:.2f}x" if base_p50 and p50 else "–"
            acc, delta = accuracy_cells(model, r, ref)
            vram = (r.get("vram") or {}).get("process_peak_mb")
            lines.append(
                f"| {name} | {BACKEND_LABELS[backend]} | {_f(p50, 2)} | "
                f"{_f(lm.get('p95_ms'), 2)} | {_f(lm.get('fps'))} | {speed} | "
                f"{_f(le.get('p50_ms'), 2)} | {_f(le.get('fps'))} | {_f(vram, 0)} | {acc} | "
                f"{delta} | {parity_cell(r)} |"
            )
    env = meta.get("env", {})
    env_line = ", ".join(
        f"{k} {env[k]}" for k in (
            "gpu", "driver", "cuda_driver", "torch", "torch_cuda", "cudnn", "onnxruntime-gpu",
            "tensorrt", "tensorrt-cu12", "cuda-python", "ultralytics",
        ) if env.get(k)
    )
    lines += [
        "",
        f"Environment: {env_line}.",
        f"Iterations: {meta.get('warmup')} warm-up + {meta.get('iters')} timed per row, batch 1. "
        f"Accuracy: YOLO on {meta.get('val_images')} BDD val images, MiDaS / YOLOP on "
        f"{meta.get('dense_frames')}; parity on {meta.get('parity_frames')} fixture frames.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- worker


def timed(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ms.append(1e3 * (time.perf_counter() - t0))
    return latency_stats(ms)


def session_outputs(session: Any, blob: np.ndarray) -> dict[str, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    outs = session.run(None, {session.get_inputs()[0].name: blob})
    return dict(zip(names, (np.asarray(o) for o in outs), strict=True))


def yolo_parts(y: np.ndarray) -> dict[str, np.ndarray]:
    return {"boxes": y[:, :4], "scores": y[:, 4:]}


def build(model: str, backend: str, cfg: dict, rc: dict) -> SimpleNamespace:
    """Live module + raw-model callables for one model and backend.

    ``raw(blob)``: what is timed as "model"; ``outputs(blob) -> {name: array}`` for parity;
    ``module(frame)``: the whole live call; ``obj``: the module object.
    """
    if model == "yolo":
        from perception.detection.detector import Detector, OnnxBackend, TensorRTBackend

        dcfg = {**rc["detector"], "backend": backend,
                "weights": str(onnx_path(cfg, "yolo").with_suffix(".pt")),
                "onnx_providers": ["CUDAExecutionProvider"]}
        if backend == "torch":
            from deploy.torch_ref import TorchYoloBackend

            be = TorchYoloBackend(dcfg, Path(cfg["weights_dir"]) / "best.pt")
        elif backend == "onnx":
            be = OnnxBackend(dcfg)
        else:
            be = TensorRTBackend(dcfg)
        det = Detector(dcfg, backend=be)
        return SimpleNamespace(
            raw=be.infer, outputs=lambda b: yolo_parts(be.infer(b)), module=det.detect,
            obj=be, providers=list(getattr(be, "providers", [])),
        )
    if model == "midas_small":
        from perception.depth.estimator import (
            DepthEstimator,
            OnnxBackend,
            TensorRTBackend,
            TorchHubBackend,
        )

        dcfg = json.loads(json.dumps(rc["depth"]))
        mcfg = dcfg["models"]["midas_small"]
        mcfg["onnx"] = str(onnx_path(cfg, "midas_small"))
        mcfg["engine"] = str(engine_path(cfg, "midas_small"))
        ecfg = {**dcfg["estimator"], "model": "midas_small", "backend": backend,
                "onnx_providers": ["CUDAExecutionProvider"], "device": "cuda", "half": False}
        dcfg["estimator"] = ecfg
        kind = {"torch": TorchHubBackend, "onnx": OnnxBackend, "tensorrt": TensorRTBackend}
        be = kind[backend](mcfg, ecfg)
        est = DepthEstimator(dcfg, backend=be)
        name = cfg["export"]["midas_output"]
        return SimpleNamespace(
            raw=be, outputs=lambda b: {name: np.asarray(be(b))}, module=est.estimate, obj=be,
            providers=[str(getattr(be, "device", backend))],
        )
    if model == "yolop":
        from perception.lanes.yolop import YolopLaneDetector, make_session

        lcfg = json.loads(json.dumps(rc["lanes"]))
        if backend == "torch":
            from deploy.torch_ref import TorchYolopSession, load_yolop

            probe = make_session(onnx_path(cfg, "yolop"), ["CPUExecutionProvider"])
            sess = TorchYolopSession(
                load_yolop(cfg["export"]["yolop_hub"], cfg["export"]["yolop_hub_model"]),
                probe.get_inputs()[0].name, tuple(o.name for o in probe.get_outputs()),
            )
        elif backend == "onnx":
            sess = make_session(onnx_path(cfg, "yolop"), ["CUDAExecutionProvider"])
        else:
            from perception.detection.runtime import TrtSession

            sess = TrtSession(engine_path(cfg, "yolop"))
        lanes = YolopLaneDetector(lcfg, session=sess)
        lane_out = lcfg["yolop"]["lane_output"]
        in_name = sess.get_inputs()[0].name
        return SimpleNamespace(
            raw=lambda b: sess.run([lane_out], {in_name: b}),
            outputs=lambda b: session_outputs(sess, b), module=lanes.lane_binary, obj=lanes,
            providers=list(sess.get_providers()),
        )
    raise KeyError(model)


def lane_mask(ll: np.ndarray, lanes: Any) -> np.ndarray:
    """Lane decision on the raw (1, 2, H, W) output with the detector's own threshold rule."""
    from perception.lanes.yolop import decision_margin

    kind = lanes.output_kind or "probs"
    margin, thr = decision_margin(np.asarray(ll, np.float32)[0], kind, lanes.threshold)
    return margin > thr


def run_accuracy(model: str, b: SimpleNamespace, cfg: dict, rc: dict, ref_dir: Path,
                 write_ref: bool, backend: str) -> dict[str, Any]:
    pairs = local_val_pairs(Path(cfg["work_dir"]) / "val")
    acc: dict[str, Any] = {}
    if model == "yolo":
        from scripts.eval_detection import gt_from_yolo

        a = cfg["accuracy"]["yolo"]
        be = b.obj
        be.conf, be.iou = float(a["conf"]), float(a["iou"])
        be.agnostic, be.max_det = bool(a["agnostic_nms"]), int(a["max_det"])
        m = MapAccumulator()
        for img_path, lbl_path in pairs:
            frame = cv2.imread(str(img_path))
            raw = be(frame)
            h, w = frame.shape[:2]
            gt = gt_from_yolo(lbl_path.read_text() if lbl_path.exists() else "", w, h)
            m.add(raw.boxes, raw.scores, raw.cls, gt.boxes, gt.cls)
        return {**m.result(), "settings": a}

    n = int(cfg["accuracy"]["dense_frames"])
    pre = blob_fn(model, rc)
    ref_file = ref_dir / f"{model}_dense.npz"
    outs = []
    for img_path, _ in pairs[:n]:
        blob = pre(cv2.imread(str(img_path)))
        if model == "midas_small":
            outs.append(np.asarray(b.raw(blob), np.float32))
        else:
            ll = b.raw(blob)[0]
            outs.append(np.packbits(lane_mask(ll, b.obj)))
    stack = np.stack(outs)
    if write_ref:
        np.savez(ref_file, out=stack)
        acc["is_reference"] = True
    if not ref_file.exists():
        return {**acc, "error": f"no reference outputs at {ref_file}"}
    ref = np.load(ref_file)["out"]
    acc["frames"] = int(len(stack))
    if model == "midas_small":
        acc["abs_rel_vs_ref"] = float(np.mean([abs_rel(r, o) for r, o in zip(ref, stack,
                                                                              strict=True)]))
        acc["max_abs_vs_ref"] = float(np.max(np.abs(ref - stack)))
    else:
        r_bits, o_bits = np.unpackbits(ref, axis=-1), np.unpackbits(stack, axis=-1)
        inter = np.logical_and(r_bits, o_bits).sum()
        union = np.logical_or(r_bits, o_bits).sum()
        acc["lane_iou_vs_ref"] = float(inter / union) if union else 1.0
        acc["lane_pixels_ref"] = int(r_bits.sum())
        acc["lane_pixels"] = int(o_bits.sum())
    return acc


def worker(cfg: dict, model: str, backend: str, result_path: Path, write_ref: bool) -> int:
    vram = VramSampler()
    res: dict[str, Any] = {"model": model, "backend": backend, "ok": False,
                           "write_ref": write_ref}
    try:
        if backend in ("torch", "onnx"):
            import torch  # noqa: F401 - loads CUDA/cuDNN libs for onnxruntime-gpu too

            res["torch_cuda"] = torch.version.cuda
            res["cudnn"] = torch.backends.cudnn.version()
        if backend == "onnx":
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        rc = runtime_configs(cfg)
        t0 = time.time()
        b = build(model, backend, cfg, rc)
        res["load_s"] = round(time.time() - t0, 1)
        res["providers"] = b.providers
        if backend == "onnx" and "CUDAExecutionProvider" not in b.providers:
            raise RuntimeError(f"ONNX Runtime fell back to {b.providers}; not a GPU number")
        pre = blob_fn(model, rc)
        pairs = local_val_pairs(Path(cfg["work_dir"]) / "val")
        idx = int(cfg["bench"]["bench_frame_index"])
        frame = cv2.imread(str(pairs[idx][0])) if pairs else fixture_frames(cfg)[0]
        res["bench_frame_hw"] = list(frame.shape[:2])
        blob = pre(frame)
        warm, iters = int(cfg["bench"]["warmup"]), int(cfg["bench"]["iters"])
        res["latency_model"] = timed(lambda: b.raw(blob), warm, iters)
        res["latency_e2e"] = timed(lambda: b.module(frame), warm, iters)
        print(model, backend, "model", res["latency_model"], "e2e", res["latency_e2e"],
              flush=True)

        ref_dir = Path(cfg["work_dir"]) / "refs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        fix = [pre(f) for f in fixture_frames(cfg)]
        outs = [b.outputs(x) for x in fix]
        fix_file = ref_dir / f"{model}_fixtures.npz"
        if write_ref:
            np.savez(fix_file, **{k: np.stack([o[k] for o in outs]) for k in outs[0]})
        elif fix_file.exists():
            ref = np.load(fix_file)
            res["parity"] = {
                k: merge_parity([parity(ref[k][i], o[k]) for i, o in enumerate(outs)])
                for k in outs[0]
            }
        res["accuracy"] = run_accuracy(model, b, cfg, rc, ref_dir, write_ref, backend)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the orchestrator reports the failed row
        traceback.print_exc()
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["vram"] = vram.stop()
    result_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0 if res["ok"] else 1


# ---------------------------------------------------------------- orchestrator


def run_worker(args: argparse.Namespace, cfg: dict, model: str, backend: str,
               write_ref: bool) -> dict[str, Any]:
    result = Path(cfg["work_dir"]) / "results" / f"{model}_{backend}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", args.config,
           "--worker", model, backend, "--result", str(result)]
    if args.smoke:
        cmd.append("--smoke")
    if write_ref:
        cmd.append("--write-ref")
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, timeout=float(cfg["bench"]["worker_timeout_s"]), check=False)
    except subprocess.TimeoutExpired:
        return {"model": model, "backend": backend, "ok": False, "error": "timeout"}
    if not result.exists():
        return {"model": model, "backend": backend, "ok": False, "error": "worker crashed"}
    res = json.loads(result.read_text(encoding="utf-8"))
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def orchestrate(args: argparse.Namespace, cfg: dict) -> int:
    t_start = time.time()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    work = Path(cfg["work_dir"])
    n_val = int(cfg["accuracy"]["val_images"])
    try:
        pairs = stage_val_subset(cfg, n_val, work / "val")
        print(f"staged {len(pairs)} BDD val images to {work / 'val'}", flush=True)
    except Exception as exc:  # noqa: BLE001 - timing still works on the fixtures
        traceback.print_exc()
        print(f"WARNING: no BDD val subset ({exc}); accuracy rows will be empty", flush=True)

    results, reference = [], {}
    for model in cfg["bench"]["models"]:
        have_ref = False
        for backend in cfg["bench"]["backends"]:
            if backend == "tensorrt" and not engine_path(cfg, model).exists():
                results.append({"model": model, "backend": backend, "ok": False,
                                "error": f"no engine {engine_path(cfg, model).name}"})
                continue
            if backend != "torch" and not onnx_path(cfg, model).exists():
                results.append({"model": model, "backend": backend, "ok": False,
                                "error": f"no ONNX {onnx_path(cfg, model).name}"})
                continue
            res = run_worker(args, cfg, model, backend, write_ref=not have_ref)
            if res.get("ok") and not have_ref:
                have_ref, reference[model] = True, backend
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("model", "backend", "ok", "error")}),
                  flush=True)

    env = {**nvidia_smi(), **package_versions()}
    for r in results:
        for k in ("torch_cuda", "cudnn"):
            if r.get(k) and not env.get(k):
                env[k] = r[k]
    meta = {
        "env": env, "warmup": cfg["bench"]["warmup"], "iters": cfg["bench"]["iters"],
        "val_images": n_val, "dense_frames": cfg["accuracy"]["dense_frames"],
        "parity_frames": cfg["parity"]["frames"], "reference": reference,
        "git_sha": git_sha(), "smoke": bool(args.smoke),
    }
    md = markdown_report(results, meta)
    print(md, flush=True)
    (out / "benchmarks.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, default=str), encoding="utf-8"
    )
    log_wandb(cfg, results, meta, out, args.smoke)
    print(f"benchmark done in {time.time() - t_start:.0f} s", flush=True)
    return 0 if all(r.get("ok") for r in results) else 1


def log_wandb(cfg: dict, results: list[dict], meta: dict, out: Path, smoke: bool) -> None:
    try:
        import wandb
    except ImportError:
        return
    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], entity=w.get("entity"), group=w["group"], job_type="benchmark",
        name="m6-bench" + ("-smoke" if smoke else ""), config={"bench": cfg, **meta},
    )
    cols = ["model", "backend", "ok", "p50_ms", "p95_ms", "fps", "e2e_p50_ms", "e2e_fps",
            "vram_mb", "accuracy", "error"]
    table = wandb.Table(columns=cols)
    for r in results:
        lm, le = r.get("latency_model") or {}, r.get("latency_e2e") or {}
        table.add_data(
            r["model"], r["backend"], bool(r.get("ok")), lm.get("p50_ms"), lm.get("p95_ms"),
            lm.get("fps"), le.get("p50_ms"), le.get("fps"),
            (r.get("vram") or {}).get("process_peak_mb"),
            json.dumps(r.get("accuracy"), default=str), r.get("error"),
        )
        key = f"{r['model']}/{r['backend']}"
        for k, v in (("p50_ms", lm.get("p50_ms")), ("p95_ms", lm.get("p95_ms")),
                     ("fps", lm.get("fps")), ("e2e_p50_ms", le.get("p50_ms")),
                     ("vram_mb", (r.get("vram") or {}).get("process_peak_mb"))):
            if v is not None:
                run.summary[f"{key}/{k}"] = v
        for k, v in (r.get("accuracy") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                run.summary[f"{key}/acc/{k}"] = v
    run.log({"bench_table": table})
    for f in ("benchmarks.md", "results.json"):
        run.save(str(out / f), base_path=str(out), policy="now")
    meta["wandb_url"] = run.url
    (out / "wandb_url.txt").write_text(run.url or "", encoding="utf-8")
    run.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("MODEL", "BACKEND"))
    ap.add_argument("--result", type=Path)
    ap.add_argument("--write-ref", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_bench_config(args.config, args.smoke)
    if args.worker:
        model, backend = args.worker
        if model not in MODELS or backend not in BACKENDS:
            ap.error(f"unknown worker {args.worker}")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        return worker(cfg, model, backend, args.result, args.write_ref)
    return orchestrate(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
