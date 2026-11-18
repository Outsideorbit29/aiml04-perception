"""M6.1: export YOLO11s, MiDaS-small and YOLOP to ONNX and check parity with PyTorch (Kaggle).

    python deploy/export.py --config configs/bench.yaml [--smoke]

Started by kaggle/export_bench/runner.py; never run on the laptop (torch, GPU). Steps:
1. YOLO11s: ``best.pt`` from the W&B artifact (``wandb.detector_artifact``), Ultralytics ONNX
   export at opset 17, static 1x3x640x640 (``configs/yolo_detector.yaml`` imgsz), simplified.
2. MiDaS-small: torch.hub ``intel-isl/MiDaS`` ``MiDaS_small``, ``torch.onnx.export`` at opset 17
   with the live input ``configs/depth.yaml`` ``models.midas_small.input_hw`` (static).
3. YOLOP: the file perception/lanes runs live, hustvl/YOLOP ``yolop-640-640.onnx``
   (``configs/lanes.yaml`` ``yolop.weights_url``, sha256-checked); its PyTorch checkpoint
   (``torch.hub`` hustvl/yolop) is the FP32 reference. It is not re-exported, so the lane
   numbers of M3 stay valid for the deployed file.
4. Parity: PyTorch FP32 vs ONNX Runtime (CUDA EP) on ``parity.frames`` fixture frames,
   preprocessed by the runtime code; max / mean abs diff per output. The ONNX input size is
   checked against the live configs.
5. Each ONNX file is logged as a W&B model artifact (``wandb.onnx_artifacts``) with its parity,
   opset, shapes and sha256. Outputs: ``models_dir``/*.onnx and ``out_dir``/export.json.
The mAP / AbsRel / lane-IoU deltas and the TensorRT parity come from deploy/benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from deploy.common import (  # noqa: E402
    MODELS,
    blob_fn,
    fixture_frames,
    git_sha,
    input_hw,
    load_bench_config,
    midas_cfg,
    nvidia_smi,
    onnx_path,
    package_versions,
    runtime_configs,
)
from deploy.metrics import merge_parity, parity  # noqa: E402


def onnx_info(path: Path) -> dict[str, Any]:
    """Opset, I/O names + shapes, size and sha256 of an ONNX file."""
    import onnx

    from perception.lanes.yolop import file_sha256

    m = onnx.load(str(path), load_external_data=False)

    def dims(vi: Any) -> list[Any]:
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]

    init = {i.name for i in m.graph.initializer}
    return {
        "file": path.name,
        "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        "producer": f"{m.producer_name} {m.producer_version}".strip(),
        "inputs": [{"name": i.name, "shape": dims(i)} for i in m.graph.input
                   if i.name not in init],
        "outputs": [{"name": o.name, "shape": dims(o)} for o in m.graph.output],
        "size_mb": round(path.stat().st_size / 2**20, 2),
        "sha256": file_sha256(path),
    }


def session_outputs(session: Any, blob: np.ndarray) -> dict[str, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    outs = session.run(None, {session.get_inputs()[0].name: blob})
    return dict(zip(names, (np.asarray(o) for o in outs), strict=True))


def yolo_parts(outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split the YOLO head (1, 4 + nc, N) into box rows (pixels) and class-score rows."""
    (y,) = outs.values()
    return {"boxes": y[:, :4], "scores": y[:, 4:]}


def check_parity(ref_fn: Any, onnx_fn: Any, blobs: list[np.ndarray]) -> dict[str, dict]:
    per: dict[str, list[dict[str, float]]] = {}
    for b in blobs:
        ref, out = ref_fn(b), onnx_fn(b)
        if set(ref) != set(out):
            raise KeyError(f"output names differ: reference {sorted(ref)} vs onnx {sorted(out)}")
        for k in ref:
            per.setdefault(k, []).append(parity(ref[k], out[k]))
    return {k: merge_parity(v) for k, v in per.items()}


def ort_session(path: Path) -> Any:
    import onnxruntime as ort

    from perception.detection.runtime import ort_providers

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    return ort.InferenceSession(
        str(path), providers=ort_providers(None, ort.get_available_providers())
    )


def assert_input_hw(info: dict[str, Any], hw: tuple[int, int], model: str) -> None:
    shape = info["inputs"][0]["shape"]
    if list(shape) != [1, 3, hw[0], hw[1]]:
        raise ValueError(f"{model}: ONNX input {shape} != live input 1x3x{hw[0]}x{hw[1]}")


# ---------------------------------------------------------------- per model


def export_yolo(cfg: dict, rc: dict, run: Any, blobs: list[np.ndarray]) -> dict[str, Any]:
    from ultralytics import YOLO

    from deploy.torch_ref import TorchYoloBackend
    from scripts.eval_detection import fetch_weights

    weights = fetch_weights(run, cfg["wandb"]["detector_artifact"], Path(cfg["weights_dir"]))
    hw = input_hw("yolo", rc)
    src = YOLO(str(weights)).export(
        format="onnx", imgsz=list(hw), opset=int(cfg["export"]["opset"]), dynamic=False,
        simplify=bool(cfg["export"]["yolo_simplify"]), half=False, batch=1, device="cpu",
    )
    dst = onnx_path(cfg, "yolo")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    info = onnx_info(dst)
    assert_input_hw(info, hw, "yolo")
    ref = TorchYoloBackend(rc["detector"], weights)
    sess = ort_session(dst)
    info["parity_onnx"] = check_parity(
        lambda b: yolo_parts({"y": ref.infer(b)}),
        lambda b: yolo_parts(session_outputs(sess, b)),
        blobs,
    )
    info["ort_providers"] = sess.get_providers()
    info["source"] = f"W&B {cfg['wandb']['detector_artifact']} (Ultralytics export)"
    info["torch_weights"] = str(weights)
    return info


def export_midas(cfg: dict, rc: dict, blobs: list[np.ndarray]) -> dict[str, Any]:
    import torch

    from deploy.torch_ref import export_onnx
    from perception.depth.estimator import TorchHubBackend

    mcfg = midas_cfg(rc)
    ref = TorchHubBackend(mcfg, {"device": "cuda", "half": False})
    hw = input_hw("midas_small", rc)
    dst = onnx_path(cfg, "midas_small")
    dummy = torch.zeros(1, 3, *hw, device=ref.device)
    export_onnx(ref.model, dummy, dst, cfg["export"]["opset"], ["image"],
                [cfg["export"]["midas_output"]])
    info = onnx_info(dst)
    assert_input_hw(info, hw, "midas_small")
    sess = ort_session(dst)
    out_name = cfg["export"]["midas_output"]
    info["parity_onnx"] = check_parity(
        lambda b: {out_name: ref(b)}, lambda b: session_outputs(sess, b), blobs
    )
    info["ort_providers"] = sess.get_providers()
    info["source"] = f"torch.hub {mcfg['hub_repo']} {mcfg['hub_model']} (torch.onnx.export)"
    return info


def export_yolop(cfg: dict, rc: dict, blobs: list[np.ndarray]) -> dict[str, Any]:
    from perception.lanes.yolop import ensure_weights

    ycfg = rc["lanes"]["yolop"]
    dst = onnx_path(cfg, "yolop")
    ensure_weights({**ycfg, "weights": str(dst)})  # download + size + sha256 check
    info = onnx_info(dst)
    assert_input_hw(info, input_hw("yolop", rc), "yolop")
    info["source"] = f"{ycfg['weights_url']} (sha256 pinned in configs/lanes.yaml)"
    sess = ort_session(dst)
    info["ort_providers"] = sess.get_providers()
    try:
        from deploy.torch_ref import TorchYolopSession, load_yolop

        model = load_yolop(cfg["export"]["yolop_hub"], cfg["export"]["yolop_hub_model"])
        ref = TorchYolopSession(
            model, sess.get_inputs()[0].name, tuple(o.name for o in sess.get_outputs())
        )
        info["parity_onnx"] = check_parity(
            lambda b: session_outputs(ref, b), lambda b: session_outputs(sess, b), blobs
        )
    except Exception as exc:  # noqa: BLE001 - the ONNX file is still valid; report the gap
        traceback.print_exc()
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def log_artifact(run: Any, cfg: dict, model: str, info: dict[str, Any]) -> str:
    import wandb

    w = cfg["wandb"]
    name = w["onnx_artifacts"][model] + (w.get("artifact_suffix") or "")
    art = wandb.Artifact(
        name, type="model",
        description=f"M6 ONNX export of {model} ({info.get('source')})",
        metadata={k: v for k, v in info.items() if k != "torch_weights"},
    )
    art.add_file(str(onnx_path(cfg, model)), name=cfg["files"][model])
    run.log_artifact(art, aliases=["m6"])
    art.wait()
    return f"{name}:{art.version}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args(argv)
    t_start = time.time()

    cfg = load_bench_config(args.config, args.smoke)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rc = runtime_configs(cfg)
    frames = fixture_frames(cfg)

    import torch  # noqa: F401 - loads the CUDA libs that onnxruntime-gpu needs
    import wandb

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], entity=w.get("entity"), group=w["group"], job_type="export",
        name="m6-export" + ("-smoke" if args.smoke else ""),
        config={"bench": cfg, "git_sha": git_sha()},
    )
    report: dict[str, Any] = {
        "git_sha": git_sha(), "smoke": args.smoke, "env": {**nvidia_smi(), **package_versions()},
        "parity_frames": len(frames), "models": {},
    }
    failed = []
    for model in args.models:
        t0 = time.time()
        blobs = [blob_fn(model, rc)(f) for f in frames]
        try:
            if model == "yolo":
                info = export_yolo(cfg, rc, run, blobs)
            elif model == "midas_small":
                info = export_midas(cfg, rc, blobs)
            else:
                info = export_yolop(cfg, rc, blobs)
            info["artifact"] = log_artifact(run, cfg, model, info)
        except Exception as exc:  # noqa: BLE001 - keep exporting the other models
            traceback.print_exc()
            info = {"error": f"{type(exc).__name__}: {exc}"}
            failed.append(model)
        info["seconds"] = round(time.time() - t0, 1)
        report["models"][model] = info
        print(model, json.dumps(info, default=str), flush=True)
        for k, p in (info.get("parity_onnx") or {}).items():
            run.summary[f"parity/{model}/{k}/max_abs"] = p["max_abs"]
            run.summary[f"parity/{model}/{k}/mean_abs"] = p["mean_abs"]

    report["seconds"] = round(time.time() - t_start, 1)
    report["wandb_url"] = run.url
    (out / "export.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    run.save(str(out / "export.json"), base_path=str(out), policy="now")
    run.finish()
    if failed:
        print(f"FAILED exports: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
