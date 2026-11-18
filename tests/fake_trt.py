"""CPU stand-ins for ``tensorrt``, cuda-python's ``cudart`` and ``onnxruntime`` (test helper).

They follow the small API surface ``perception.detection.runtime`` uses, with real host memory:
``cudaMallocHost`` hands out ctypes buffers, device buffers are numpy arrays keyed by a fake
pointer, and memcpys move bytes between them. ``FakeEngine`` computes each output from the first
input with a per-output function, so a ``TrtRunner`` round trip can be checked numerically.

    install(monkeypatch, engines={"m.engine": FakeEngine(...)})   # patches sys.modules + _cudart
"""

from __future__ import annotations

import ctypes
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np


# ---------------------------------------------------------------- cudart
class FakeStream:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __int__(self) -> int:
        return self.handle


class FakeCudart:
    """Records calls; device memory = dict ptr -> uint8 array; host memory = ctypes buffers."""

    cudaMemcpyKind = SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2)

    def __init__(self) -> None:
        self.device: dict[int, np.ndarray] = {}
        self.host: dict[int, Any] = {}
        self.calls: list[tuple] = []
        self.streams: set[int] = set()
        self._next = 0x1000

    def _ptr(self) -> int:
        self._next += 0x1000
        return self._next

    def cudaMalloc(self, nbytes: int) -> tuple[int, int]:
        p = self._ptr()
        self.device[p] = np.zeros(nbytes, np.uint8)
        return 0, p

    def cudaMallocHost(self, nbytes: int) -> tuple[int, int]:
        buf = (ctypes.c_uint8 * nbytes)()
        addr = ctypes.addressof(buf)
        self.host[addr] = buf
        return 0, addr

    def cudaFree(self, p: int) -> tuple[int]:
        self.calls.append(("free", p))
        del self.device[p]
        return (0,)

    def cudaFreeHost(self, p: int) -> tuple[int]:
        self.calls.append(("free_host", p))
        del self.host[p]
        return (0,)

    def cudaStreamCreate(self) -> tuple[int, FakeStream]:
        s = FakeStream(self._ptr())
        self.streams.add(int(s))
        return 0, s

    def cudaStreamDestroy(self, s: FakeStream) -> tuple[int]:
        self.calls.append(("stream_destroy", int(s)))
        self.streams.discard(int(s))
        return (0,)

    def cudaStreamSynchronize(self, s: FakeStream) -> tuple[int]:
        self.calls.append(("sync", int(s)))
        return (0,)

    def cudaMemcpyAsync(self, dst: int, src: int, nbytes: int, kind: int, s: FakeStream):
        if kind == self.cudaMemcpyKind.cudaMemcpyHostToDevice:
            self.calls.append(("h2d", dst))
            self.device[dst][:nbytes] = np.frombuffer(
                (ctypes.c_uint8 * nbytes).from_address(src), np.uint8
            )
        else:
            self.calls.append(("d2h", src))
            ctypes.memmove(dst, self.device[src].ctypes.data, nbytes)
        return (0,)


# ---------------------------------------------------------------- tensorrt
class TensorIOMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class FakeEngine:
    """``io``: name -> (shape, numpy dtype, mode); ``fns``: output name -> f(first input)."""

    def __init__(self, io: dict[str, tuple], fns: dict[str, Callable[[np.ndarray], Any]]):
        self.io = io
        self.fns = fns
        self.cudart: FakeCudart | None = None

    @property
    def num_io_tensors(self) -> int:
        return len(self.io)

    def get_tensor_name(self, i: int) -> str:
        return list(self.io)[i]

    def get_tensor_shape(self, name: str) -> tuple:
        return self.io[name][0]

    def get_tensor_dtype(self, name: str) -> Any:
        return self.io[name][1]

    def get_tensor_mode(self, name: str) -> str:
        return self.io[name][2]

    def create_execution_context(self) -> FakeContext:
        return FakeContext(self)


class FakeContext:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.addr: dict[str, int] = {}

    def set_tensor_address(self, name: str, ptr: int) -> None:
        self.addr[name] = ptr

    def _view(self, name: str) -> np.ndarray:
        shape, dtype, _ = self.engine.io[name]
        cr = self.engine.cudart
        assert cr is not None
        return cr.device[self.addr[name]].view(np.dtype(dtype)).reshape(shape)

    def execute_async_v3(self, stream: int) -> bool:
        cr = self.engine.cudart
        assert cr is not None and isinstance(stream, int) and stream in cr.streams
        cr.calls.append(("execute", stream))
        inputs = [n for n, v in self.engine.io.items() if v[2] == TensorIOMode.INPUT]
        x = self._view(inputs[0]).copy()
        for name, fn in self.engine.fns.items():
            self._view(name)[...] = fn(x)
        return True


def make_tensorrt(engines: dict[str, FakeEngine], cudart: FakeCudart) -> types.ModuleType:
    """A ``tensorrt`` module whose Runtime deserializes engine files by their text content."""
    trt = types.ModuleType("tensorrt")
    trt.__version__ = "10.0.fake"
    trt.TensorIOMode = TensorIOMode

    class Logger:
        WARNING = 2

        def __init__(self, level: int) -> None:
            self.level = level

    class Runtime:
        def __init__(self, logger: Logger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, data: bytes) -> FakeEngine | None:
            eng = engines.get(data.decode())
            if eng is not None:
                eng.cudart = cudart
            return eng

    trt.Logger = Logger
    trt.Runtime = Runtime
    trt.nptype = lambda dt: np.dtype(dt).type
    return trt


# ---------------------------------------------------------------- onnxruntime
def make_onnxruntime(
    input_shape: list, outputs: dict[str, Callable[[np.ndarray], Any]], meta: dict | None = None
) -> types.ModuleType:
    """An ``onnxruntime`` module with CPU only and one session layout for every file."""
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]

    class InferenceSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            self.path, self.providers = path, providers

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="images", shape=list(input_shape))]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=n, shape=None) for n in outputs]

        def get_providers(self) -> list[str]:
            return list(self.providers)

        def get_modelmeta(self) -> SimpleNamespace:
            return SimpleNamespace(custom_metadata_map=dict(meta or {}))

        def run(self, names: list[str] | None, feed: dict[str, np.ndarray]) -> list:
            x = feed["images"]
            return [outputs[n](x) for n in (names or list(outputs))]

    ort.InferenceSession = InferenceSession
    return ort


def install(monkeypatch: Any, engines: dict[str, FakeEngine]) -> FakeCudart:
    """Patch ``tensorrt`` into sys.modules and the runtime's cudart loader; returns the cudart."""
    import perception.detection.runtime as runtime

    cudart = FakeCudart()
    monkeypatch.setitem(sys.modules, "tensorrt", make_tensorrt(engines, cudart))
    monkeypatch.setattr(runtime, "_cudart", lambda: cudart)
    return cudart


def yolo_head(boxes_cxcywh: np.ndarray, cls: np.ndarray, scores: np.ndarray, nc: int, n: int):
    """A (1, 4 + nc, n) YOLO11 head with the given candidates first and zeros after them."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    k = len(boxes_cxcywh)
    out[0, :4, :k] = np.asarray(boxes_cxcywh, np.float32).T
    out[0, 4 + np.asarray(cls), np.arange(k)] = scores
    return out
