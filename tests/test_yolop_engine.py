"""YOLOP loads a TensorRT engine when the weights path ends in .engine (M6.4)."""

import perception.detection.runtime as runtime
from perception.lanes.yolop import make_session


def test_make_session_uses_trt_for_engine_files(monkeypatch):
    seen = {}

    class FakeTrt:
        def __init__(self, path):
            seen["path"] = str(path)

    monkeypatch.setattr(runtime, "TrtSession", FakeTrt)
    sess = make_session("models/yolop-640-640.engine", ["CUDAExecutionProvider"])
    assert isinstance(sess, FakeTrt)
    assert seen["path"] == "models/yolop-640-640.engine"
