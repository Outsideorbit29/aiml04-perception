"""Build a TensorRT engine from an ONNX file (tensorrt Python API, no torch, no trtexec).

Equivalent to ``trtexec --onnx=<m>.onnx --saveEngine=<m>.engine --fp16
--memPoolSize=workspace:<MiB>`` with the default builder optimisation level. Used on the Kaggle
T4 (M6.2, through ``deploy/build_local_engine.py``) and on the laptop (M6.4): engines only run
on the GPU model and TensorRT version they were built with, so every machine builds its own.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

MIB = 1 << 20


def engine_path_for(onnx_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """``<out_dir or the ONNX folder>/<stem>.engine``: the name the runtime backends look for."""
    p = Path(onnx_path)
    return Path(out_dir or p.parent) / f"{p.stem}.engine"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    workspace_mib: int = 1024,
    verbose: bool = False,
) -> dict[str, Any]:
    """Parse ``onnx_path``, build a serialized engine and write it to ``engine_path``.

    Returns a summary dict (also written next to the engine as ``<stem>.engine.json``).
    """
    import tensorrt as trt  # lazy: only on machines that build engines

    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    major = int(trt.__version__.split(".")[0])
    # TensorRT 10 networks are always explicit-batch; 8.x needs the flag.
    flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    for i in range(network.num_inputs):
        shape = tuple(network.get_input(i).shape)
        if any(d < 0 for d in shape):
            raise ValueError(f"{onnx_path}: input {i} has a dynamic shape {shape}; export static")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * MIB)
    use_fp16 = bool(fp16)
    if use_fp16 and not builder.platform_has_fast_fp16:
        print(f"WARNING: no fast FP16 on this GPU; building {onnx_path.name} in FP32", flush=True)
        use_fp16 = False
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": "fp16" if use_fp16 else "fp32",
        "workspace_mib": int(workspace_mib),
        "build_s": round(time.time() - t0, 1),
        "engine_mb": round(engine_path.stat().st_size / MIB, 2),
        "tensorrt": trt.__version__,
        "gpu": gpu_name(),
        "inputs": [
            {"name": network.get_input(i).name, "shape": list(network.get_input(i).shape)}
            for i in range(network.num_inputs)
        ],
        "outputs": [
            {"name": network.get_output(i).name, "shape": list(network.get_output(i).shape)}
            for i in range(network.num_outputs)
        ],
    }
    Path(str(engine_path) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
