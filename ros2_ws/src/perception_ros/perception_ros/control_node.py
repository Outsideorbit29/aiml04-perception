"""control_node (M5.4): /perception/objects + /carla/ego/odom -> /carla/ego/control.

A thin wrapper around ``perception.control.LongitudinalController`` (AEB / ACC / gap rules with
lead persistence; thresholds in ``control`` of configs/carla.yaml). One EgoControl is published
per objects message, with the same header stamp, which is what the bridge waits for before it
ticks the simulator.
"""

from __future__ import annotations

import time

import rclpy
from nav_msgs.msg import Odometry
from perception_msgs.msg import EgoControl, TrackedObjectArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from perception_ros.conversions import candidates_from_msg, fill_control
from perception_ros.repo import ensure_repo_on_path

ensure_repo_on_path()
from perception.control import LongitudinalController  # noqa: E402
from sim.carla_utils import load_carla_config  # noqa: E402


class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__("control_node")
        self.declare_parameter("config", "configs/carla.yaml")
        cfg = load_carla_config(self.get_parameter("config").value)
        self.cfg = cfg["control"]
        self.ctrl = LongitudinalController(self.cfg)
        self.speed = 0.0
        self.t_speed = -1.0
        self.last_mode = ""
        self.pub = self.create_publisher(EgoControl, "/carla/ego/control", 10)
        self.create_subscription(TrackedObjectArray, "/perception/objects", self._on_objects, 10)
        self.create_subscription(Odometry, "/carla/ego/odom", self._on_odom, 10)
        self.get_logger().info(
            f"ready: lead persistence {self.cfg['lead_persist_frames']} frames, AEB ttc < "
            f"{self.cfg['aeb_ttc_s']} s or d < {self.cfg['aeb_distance_m']} m"
        )

    def _on_odom(self, msg: Odometry) -> None:
        self.speed = float(msg.twist.twist.linear.x)
        self.t_speed = time.monotonic()

    def _on_objects(self, msg: TrackedObjectArray) -> None:
        fresh = 0 <= time.monotonic() - self.t_speed <= float(self.cfg["odom_timeout_s"])
        speed = self.speed if fresh and self.t_speed >= 0 else 0.0
        decision = self.ctrl.update(candidates_from_msg(msg), speed)
        out = EgoControl()
        out.header = msg.header
        fill_control(out, decision, speed)
        self.pub.publish(out)
        if decision.mode != self.last_mode:
            self.get_logger().info(
                f"{self.last_mode or 'START'} -> {decision.mode} ({decision.reason}), "
                f"speed {speed:.1f} m/s, brake {decision.brake:.2f}"
            )
            self.last_mode = decision.mode


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
