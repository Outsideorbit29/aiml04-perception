"""perception_node (M5.3): /carla/ego/rgb -> perception.pipeline -> objects, lanes, debug image.

A thin wrapper: converts the Image to numpy, calls ``Pipeline.process`` (detector, ByteTrack,
lanes, MiDaS, DistanceFusion, all from perception/) and publishes the result. Models run on ONNX
Runtime (``perception.backend`` in configs/carla.yaml; TensorRT after M6.4), never PyTorch.

Topics: in ``/carla/ego/rgb``; out ``/perception/objects`` (TrackedObjectArray),
``/perception/lanes`` (LaneBoundaries), ``/perception/debug_image`` (bgr8, annotated). Output
headers copy the input image header, so control_node and the bridge can match frames by stamp.
"""

from __future__ import annotations

import time

import rclpy
from perception_msgs.msg import LaneBoundaries, TrackedObject, TrackedObjectArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from perception_ros.conversions import (
    fill_image,
    fill_lanes,
    fill_object_array,
    image_to_bgr,
    stamp_to_ns,
)
from perception_ros.repo import ensure_repo_on_path

ensure_repo_on_path()
from perception.pipeline import (  # noqa: E402
    Pipeline,
    draw_frame,
    frame_to_dict,
    load_pipeline_config,
)
from sim.carla_utils import load_carla_config  # noqa: E402


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("config", "configs/carla.yaml")
        cfg = load_carla_config(self.get_parameter("config").value)
        p = cfg["perception"]
        self.pcfg = load_pipeline_config(p["pipeline_config"])
        log = self.get_logger()
        log.info(f"loading pipeline {p['pipeline_config']} (backend {p['backend']})")
        t0 = time.monotonic()
        try:
            self.pipe = Pipeline.from_config(
                self.pcfg,
                backend=p["backend"],
                camera_name=p["camera"],
                fps=float(p["fps"]),
                detector_weights=p.get("detector_weights"),
                yolop_weights=p.get("yolop_weights"),
            )
        except (FileNotFoundError, NotImplementedError) as e:
            log.error(
                f"{type(e).__name__}: {e}\nThe live pipeline needs the M6 ONNX exports in "
                "models/ (yolo11s-bdd.onnx, midas_small.onnx, yolop-640-640.onnx); paths are in "
                "configs/pipeline_carla.yaml and configs/carla.yaml `perception`."
            )
            raise
        log.info(f"pipeline ready in {time.monotonic() - t0:.1f} s: {self.pipe.describe()}")
        self.publish_debug = bool(p["publish_debug"])
        self.debug_every = max(1, int(p["debug_every_n"]))
        self.lane_viz = self.pipe.stage_cfgs["lanes"].get("viz")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_obj = self.create_publisher(TrackedObjectArray, "/perception/objects", 10)
        self.pub_lanes = self.create_publisher(LaneBoundaries, "/perception/lanes", 10)
        self.pub_dbg = self.create_publisher(Image, "/perception/debug_image", qos)
        self.create_subscription(Image, "/carla/ego/rgb", self._on_image, qos)
        self.n = 0
        self.t_sum = 0.0

    def _on_image(self, msg: Image) -> None:
        img = image_to_bgr(msg)
        pf = self.pipe.process(img, stamp=stamp_to_ns(msg.header.stamp) * 1e-9, frame_id=self.n)
        d = frame_to_dict(pf, self.pipe.last_stats)
        out = TrackedObjectArray()
        out.header = msg.header
        fill_object_array(out, d, TrackedObject)
        self.pub_obj.publish(out)
        lanes = LaneBoundaries()
        lanes.header = msg.header
        fill_lanes(lanes, d["lanes"], self.pipe.lane_debug())
        self.pub_lanes.publish(lanes)
        if self.publish_debug and self.n % self.debug_every == 0:
            vis = draw_frame(
                img, pf, self.pcfg["viz"], self.pipe.lane_debug(), self.pipe.last_tracks,
                self.lane_viz,
            )
            dbg = Image()
            dbg.header = msg.header
            self.pub_dbg.publish(fill_image(dbg, vis, "bgr8"))
        self.n += 1
        self.t_sum += pf.timings_ms.get("total_ms", 0.0)
        if self.n % 100 == 0:
            self.get_logger().info(
                f"{self.n} frames, mean pipeline {self.t_sum / 100:.1f} ms, "
                f"{len(pf.objects)} objects, lead {d['lead_track_id']}"
            )
            self.t_sum = 0.0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
