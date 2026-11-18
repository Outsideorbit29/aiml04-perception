"""recorder_node (M5.7): /perception/debug_image -> annotated MP4 in media/ (gitignored).

One video frame per debug image (one per sim step), written at ``recorder.fps`` so the video
plays at sim speed even when the loop ran slower than real time. With
``recorder.overlay_control`` the latest /carla/ego/control (mode, brake, ego speed, lead) is
drawn in the bottom-left corner. rosbag2 recording is a separate process started by
stack.launch.py (``record_bag:=true``).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import rclpy
from perception_msgs.msg import EgoControl
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from perception_ros.conversions import image_to_bgr
from perception_ros.repo import ensure_repo_on_path

ROOT = ensure_repo_on_path()
from sim.carla_utils import load_carla_config  # noqa: E402

MODE_COLORS = {"AEB": (0, 0, 255), "HOLD": (0, 0, 200), "ACC": (0, 165, 255), "GAP": (0, 200, 255)}


class RecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("recorder_node")
        self.declare_parameter("config", "configs/carla.yaml")
        self.declare_parameter("name", "stack")
        cfg = load_carla_config(self.get_parameter("config").value)["recorder"]
        self.cfg = cfg
        out_dir = Path(cfg["out_dir"])
        out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self.get_parameter("name").value or "stack"
        self.path = out_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.writer: cv2.VideoWriter | None = None
        self.frames = 0
        self.control: EgoControl | None = None
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, "/perception/debug_image", self._on_image, qos)
        self.create_subscription(EgoControl, "/carla/ego/control", self._on_control, 10)

    def _on_control(self, msg: EgoControl) -> None:
        self.control = msg

    def _on_image(self, msg: Image) -> None:
        img = image_to_bgr(msg)
        if self.cfg.get("overlay_control") and self.control is not None:
            c = self.control
            txt = f"{c.mode}  brake {c.brake:.2f}  ego {c.ego_speed_mps * 3.6:.0f} km/h"
            if c.lead_track_id >= 0:
                txt += f"  lead #{c.lead_track_id} {c.lead_distance_m:.1f} m"
            color = MODE_COLORS.get(c.mode, (0, 160, 0))
            h = img.shape[0]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (0, h - th - 14), (tw + 12, h), color, -1)
            cv2.putText(img, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*str(self.cfg["fourcc"]))
            self.writer = cv2.VideoWriter(
                str(self.path), fourcc, float(self.cfg["fps"]), (img.shape[1], img.shape[0])
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {self.path}")
            self.get_logger().info(f"recording to {self.path}")
        self.writer.write(img)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"wrote {self.frames} frames to {self.path}")
            self.writer = None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
