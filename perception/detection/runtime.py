"""Exported-model runtimes shared by the detector, depth and lane backends (M6, no torch).

- ``ort_providers``: onnxruntime providers to request. Defaults to CUDA then CPU and never the
  TensorRT execution provider, so an ``onnx`` backend is plain ONNX Runtime (onnxruntime-gpu
  lists the TensorRT EP as available even when it would build engines behind our back).
- ``TrtRunner``: a deserialized TensorRT engine with static I/O shapes, numpy in / numpy out.
  Uses ``tensorrt`` and ``cuda-python`` only, so it runs in the laptop runtime (WSL2, no
  PyTorch). Host buffers are pinned and reused; outputs are returned as copies.
- ``TrtSession``: ``TrtRunner`` behind the small part of the ``onnxruntime.InferenceSession`` API
  that ``perception.lanes.yolop.YolopLaneDetector`` uses (``get_inputs``, ``run``), so the lane
  backend can take a TensorRT engine through its ``session`` argument unchanged.

Engines only run on the GPU model and TensorRT version they were built with:
build them with ``deploy/build_local_engine.py`` on the machine that runs them.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

DEFAULT_ORT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def ort_providers(requested: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    """Requested providers that are available, in order; CUDA then CPU when none are given.

    Falls back to CPU so a session can always be created (``describe()`` / the providers list
    shows the fallback).
    """
    wanted = list(requested) if requested else list(DEFAULT_ORT_PROVIDERS)
    chosen = [p for p in wanted if p in set(available)]
    return chosen or ["CPUExecutionProvider"]


def static_shape(shape: Sequence[Any]) -> tuple[int, ...] | None:
    """``(1, 3, 640, 640)`` if every dimension is a fixed positive int, else None (dynamic)."""
    if not all(isinstance(d, (int, np.integer)) and int(d) > 0 for d in shape):
        return None
    return tuple(int(d) for d in shape)


def check_input_hw(shape: Sequence[Any], hw: tuple[int, int], what: str) -> None:
    """Raise if a static NCHW model input does not take ``hw`` (dynamic inputs pass)."""
    s = static_shape(shape)
    if s is not None and len(s) == 4 and s[2:] != tuple(hw):
        raise ValueError(
            f"{what} takes a fixed {s[2]}x{s[3]} input but the config asks for {hw[0]}x{hw[1]}; "
            "re-export the model at that size (deploy/export.py) or fix the config"
        )


def _cudart() -> Any:
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
    except ImportError:
        from cuda import cudart  # older cuda-python 12.x
    return cudart


def _check(ret: Any) -> Any:
    """cuda-python calls return ``(err, *values)``; raise on error, return the value(s)."""
    err, *vals = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime error {err}")
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(vals)


class _Tensor:
    """One engine I/O tensor: pinned host array + device buffer of the same size."""

    def __init__(self, cudart: Any, name: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.name, self.shape, self.dtype = name, shape, dtype
        self.nbytes = int(np.prod(shape)) * dtype.itemsize
        self.device = int(_check(cudart.cudaMalloc(self.nbytes)))
        self.host_ptr = int(_check(cudart.cudaMallocHost(self.nbytes)))
        raw = (ctypes.c_uint8 * self.nbytes).from_address(self.host_ptr)
        self.host = np.frombuffer(raw, dtype=dtype).reshape(shape)


class TrtRunner:
    """TensorRT engine with static shapes: ``runner(x0, x1, ...) -> [out0, out1, ...]``.

    Inputs are cast to the engine's input dtype and must have its exact shape.
    """

    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt  # lazy: only the tensorrt backend needs it

        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found at {path}; build it on this machine with "
                "deploy/build_local_engine.py"
            )
        self.trt = trt
        self.cudart = _cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"could not deserialize {path}: engines only load on the GPU model and TensorRT "
                f"version ({trt.__version__} here) they were built with; rebuild it with "
                "deploy/build_local_engine.py"
            )
        self.context = self.engine.create_execution_context()
        self.stream = _check(self.cudart.cudaStreamCreate())
        self.inputs: list[_Tensor] = []
        self.outputs: list[_Tensor] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = static_shape(tuple(self.engine.get_tensor_shape(name)))
            if shape is None:
                raise ValueError(
                    f"{path}: tensor {name} has a dynamic shape; export the ONNX model with "
                    "static shapes"
                )
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            t = _Tensor(self.cudart, name, shape, dtype)
            self.context.set_tensor_address(name, t.device)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs).append(t)
        self.path = path

    @property
    def input_names(self) -> list[str]:
        return [t.name for t in self.inputs]

    @property
    def output_names(self) -> list[str]:
        return [t.name for t in self.outputs]

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        if len(arrays) != len(self.inputs):
            raise ValueError(f"engine takes {len(self.inputs)} inputs, got {len(arrays)}")
        cr = self.cudart
        h2d = cr.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cr.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for t, x in zip(self.inputs, arrays, strict=True):
            x = np.asarray(x)
            if x.shape != t.shape:
                raise ValueError(f"input {t.name}: expected shape {t.shape}, got {x.shape}")
            np.copyto(t.host, x, casting="same_kind")
            _check(cr.cudaMemcpyAsync(t.device, t.host_ptr, t.nbytes, h2d, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError(f"TensorRT execution failed for {self.path}")
        for t in self.outputs:
            _check(cr.cudaMemcpyAsync(t.host_ptr, t.device, t.nbytes, d2h, self.stream))
        _check(cr.cudaStreamSynchronize(self.stream))
        return [t.host.copy() for t in self.outputs]

    def close(self) -> None:
        """Free the CUDA buffers (also done at interpreter exit by the driver)."""
        cr = self.cudart
        for t in self.inputs + self.outputs:
            cr.cudaFree(t.device)
            cr.cudaFreeHost(t.host_ptr)
        self.inputs, self.outputs = [], []
        cr.cudaStreamDestroy(self.stream)


class TrtSession:
    """``onnxruntime.InferenceSession``-like wrapper of ``TrtRunner`` (``get_inputs``,
    ``get_outputs``, ``get_providers``, ``run(output_names, {input_name: array})``)."""

    def __init__(self, engine_path: str | Path, runner: Any | None = None) -> None:
        self.runner = runner if runner is not None else TrtRunner(engine_path)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=t.name, shape=list(t.shape)) for t in self.runner.inputs]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=t.name, shape=list(t.shape)) for t in self.runner.outputs]

    def get_providers(self) -> list[str]:
        return ["TensorRT"]

    def run(self, output_names: Sequence[str] | None, feed: dict[str, np.ndarray]) -> list:
        missing = [n for n in self.runner.input_names if n not in feed]
        if missing:
            raise KeyError(f"missing inputs {missing}")
        outs = dict(
            zip(
                self.runner.output_names,
                self.runner(*(feed[n] for n in self.runner.input_names)),
                strict=True,
            )
        )
        names = list(output_names) if output_names else self.runner.output_names
        return [outs[n] for n in names]
