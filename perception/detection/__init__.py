"""Object detection (M2.4): ``Detector(cfg).detect(frame) -> list[Detection]``."""

from perception.detection.detector import BACKENDS, Detector, ExportedBackend, load_config

__all__ = ["BACKENDS", "Detector", "ExportedBackend", "load_config"]
