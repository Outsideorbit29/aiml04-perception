"""carla_bridge_node (M5.2): the only node that talks to CARLA and the only one that ticks it.

Per sim step (synchronous mode, ``carla.fixed_delta_seconds`` = 0.05 s):
1. ``world.tick()``; wait for that frame's RGB (and depth GT) camera data.
2. Publish ``/carla/ego/rgb`` (bgr8), ``/carla/ego/depth_gt`` (32FC1, metres) and
   ``/carla/ego/odom`` with ``header.stamp`` = sim time of the frame.
3. Wait for ``/carla/ego/control`` stamped with that frame (control_node copies the stamp), at
   most ``bridge.control_timeout_s`` (``first_control_timeout_s`` before the first one, while
   the ONNX models load). Then apply it and tick again, so perception latency never desyncs the
   simulation.

Lateral control is the Traffic Manager autopilot. ``override_autopilot`` controls switch the
autopilot off and apply throttle/brake with the autopilot's last steering angle; a
non-override control switches the autopilot back on. The TM is told to ignore vehicles,
walkers, lights and signs (configs/carla.yaml ``traffic_manager``), so only perception stops
the car.

On shutdown (Ctrl+C, launch shutdown, error) the sensors and the ego are destroyed and the
world and TM go back to asynchronous mode, so the server is left clean.
"""

from __future__ import annotations

import math
import time
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from perception_msgs.msg import EgoControl
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from perception_ros.conversions import fill_image, set_stamp_ns, stamp_to_ns
from perception_ros.repo import ensure_repo_on_path

ensure_repo_on_path()
from sim import carla_utils as cu  # noqa: E402


class CarlaBridge(Node):
    def __init__(self) -> None:
        super().__init__("carla_bridge_node")
        self.declare_parameter("config", "configs/carla.yaml")
        self.declare_parameter("carla_host", "")
        self.declare_parameter("carla_port", 0)
        self.declare_parameter("town", "")
        self.declare_parameter("load_town", False)
        gp = self.get_parameter
        self.cfg = cu.load_carla_config(gp("config").value)
        c = self.cfg["carla"]
        self.host = gp("carla_host").value or c["host"]
        self.port = int(gp("carla_port").value or c["port"])
        self.town = gp("town").value or c["town"]
        self.load_town = bool(gp("load_town").value or c["load_town"])
        b = self.cfg["bridge"]
        self.bcfg = b
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_rgb = self.create_publisher(Image, "/carla/ego/rgb", qos)
        self.pub_depth = self.create_publisher(Image, "/carla/ego/depth_gt", qos)
        self.pub_odom = self.create_publisher(Odometry, "/carla/ego/odom", 10)
        self.create_subscription(EgoControl, "/carla/ego/control", self._on_control, 10)
        self.waiter = cu.ControlWaiter(b["control_timeout_s"], b["first_control_timeout_s"])
        self.last_control: EgoControl | None = None
        self.client: Any = None
        self.world: Any = None
        self.tm: Any = None
        self.ego: Any = None
        self.actors: list[Any] = []
        self.original_settings: Any = None
        self.autopilot_on = False
        self.q_rgb = cu.SensorQueue()
        self.q_depth = cu.SensorQueue()

    # ------------------------------------------------------------------ setup / teardown
    def setup(self) -> None:
        c = self.cfg["carla"]
        self.get_logger().info(f"connecting to CARLA at {self.host}:{self.port}")
        self.client = cu.connect(self.host, self.port, float(c["timeout_s"]))
        self.world = cu.ensure_town(self.client, self.town, self.load_town)
        self.get_logger().info(f"map {self.world.get_map().name}")
        self.original_settings = cu.set_synchronous(self.world, float(c["fixed_delta_seconds"]))
        self.tm = self.client.get_trafficmanager(int(self.cfg["traffic_manager"]["port"]))
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(int(c["seed"]))
        self.ego = cu.spawn_ego(self.world, self.cfg["ego"])
        self.actors.append(self.ego)
        bp_lib = self.world.get_blueprint_library()
        cam_tf = cu.camera_transform(self.cfg["camera"])
        rgb = self.world.spawn_actor(
            cu.camera_blueprint(bp_lib, self.cfg["camera"], "rgb"), cam_tf, attach_to=self.ego
        )
        self.actors.append(rgb)
        rgb.listen(self.q_rgb)
        self.depth_enabled = bool(self.cfg["depth_camera"]["enabled"])
        if self.depth_enabled:
            dep = self.world.spawn_actor(
                cu.camera_blueprint(bp_lib, self.cfg["camera"], "depth"), cam_tf,
                attach_to=self.ego,
            )
            self.actors.append(dep)
            dep.listen(self.q_depth)
        self._configure_tm()
        self.world.tick(float(c["tick_timeout_s"]))  # actors become valid
        self.get_logger().info(f"ego {self.ego.type_id} id {self.ego.id} spawned")

    def _configure_tm(self) -> None:
        t = self.cfg["traffic_manager"]
        tm, ego = self.tm, self.ego
        ego.set_autopilot(True, tm.get_port())
        self.autopilot_on = True
        tm.ignore_vehicles_percentage(ego, float(t["ignore_vehicles_pct"]))
        tm.ignore_walkers_percentage(ego, float(t["ignore_walkers_pct"]))
        tm.ignore_lights_percentage(ego, float(t["ignore_lights_pct"]))
        tm.ignore_signs_percentage(ego, float(t["ignore_signs_pct"]))
        tm.auto_lane_change(ego, bool(t["auto_lane_change"]))
        self._try_tm("set_desired_speed", ego, float(t["desired_speed_kmh"]))
        if t.get("route"):
            self._try_tm("set_route", ego, list(t["route"]))
        if t.get("update_vehicle_lights"):
            self._try_tm("update_vehicle_lights", ego, True)

    def _try_tm(self, name: str, *args: Any) -> None:
        """Optional Traffic Manager feature: log instead of failing the whole bridge."""
        try:
            getattr(self.tm, name)(*args)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"traffic manager {name} failed: {e}")

    def teardown(self) -> None:
        log = self.get_logger()
        if self.ego is not None and self.autopilot_on:
            try:
                self.ego.set_autopilot(False, self.tm.get_port())
            except Exception:  # noqa: BLE001
                pass
        if self.client is not None:
            n = cu.destroy_actors(self.client, self.actors)
            log.info(f"destroyed {n}/{len(self.actors)} actors")
        self.actors = []
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:  # noqa: BLE001
                pass
        if self.world is not None:
            cu.restore_settings(self.world, self.original_settings)
            log.info("world settings restored (asynchronous mode)")

    # ------------------------------------------------------------------ loop
    def _on_control(self, msg: EgoControl) -> None:
        self.last_control = msg
        self.waiter.offer(stamp_to_ns(msg.header.stamp))

    def run(self) -> None:
        c, b = self.cfg["carla"], self.bcfg
        max_frames = int(b["max_frames"])
        n = 0
        t_log = time.monotonic()
        while rclpy.ok() and (max_frames <= 0 or n < max_frames):
            frame = self.world.tick(float(c["tick_timeout_s"]))
            rgb = self.q_rgb.get(frame, float(b["sensor_timeout_s"]))
            depth = self.q_depth.get(frame, float(b["sensor_timeout_s"])) if self.depth_enabled \
                else None
            stamp_ns = int(round(rgb.timestamp * 1e9))
            self._publish(rgb, depth, stamp_ns)
            timeout = self.waiter.expect(stamp_ns)
            t_end = time.monotonic() + timeout
            while rclpy.ok() and not self.waiter.got_expected:
                left = t_end - time.monotonic()
                if left <= 0:
                    break
                rclpy.spin_once(self, timeout_sec=min(0.02, left))
            late = not self.waiter.got_expected
            if late:
                self.waiter.timed_out()
            self._apply(self.last_control, late)
            n += 1
            if int(b["log_every"]) and n % int(b["log_every"]) == 0:
                dt = time.monotonic() - t_log
                t_log = time.monotonic()
                mode = self.last_control.mode if self.last_control is not None else "-"
                self.get_logger().info(
                    f"frame {n}: {int(b['log_every']) / dt:.1f} sim steps/s, control "
                    f"timeouts {self.waiter.timeouts}, mode {mode}"
                )

    def _publish(self, rgb: Any, depth: Any, stamp_ns: int) -> None:
        fid = str(self.bcfg["frame_id"])
        img = Image()
        set_stamp_ns(img.header.stamp, stamp_ns)
        img.header.frame_id = fid
        fill_image(img, cu.bgra_to_bgr(rgb.raw_data, rgb.height, rgb.width), "bgr8")
        self.pub_rgb.publish(img)
        if depth is not None:
            dm = Image()
            dm.header = img.header
            fill_image(dm, cu.decode_depth(depth.raw_data, depth.height, depth.width), "32FC1")
            self.pub_depth.publish(dm)
        tf = self.ego.get_transform()
        v = self.ego.get_velocity()
        av = self.ego.get_angular_velocity()
        pos, q = cu.carla_to_ros_pose(
            tf.location.x, tf.location.y, tf.location.z,
            tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw,
        )
        od = Odometry()
        od.header.stamp = img.header.stamp
        od.header.frame_id = str(self.bcfg["odom_frame"])
        od.child_frame_id = str(self.bcfg["base_frame"])
        p = od.pose.pose
        p.position.x, p.position.y, p.position.z = pos
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        od.twist.twist.linear.x = cu.forward_speed(v.x, v.y, tf.rotation.yaw)
        od.twist.twist.angular.z = -math.radians(av.z)
        self.pub_odom.publish(od)

    def _apply(self, msg: EgoControl | None, late: bool) -> None:
        import carla

        override = msg is not None and bool(msg.override_autopilot)
        if late and self.bcfg["on_control_timeout"] == "autopilot":
            override = False
        port = self.tm.get_port()
        if not override:
            if not self.autopilot_on:
                self.ego.set_autopilot(True, port)
                self.autopilot_on = True
            return
        steer = float(self.ego.get_control().steer)  # the autopilot's last steering angle
        if self.autopilot_on:
            self.ego.set_autopilot(False, port)
            self.autopilot_on = False
        self.ego.apply_control(
            carla.VehicleControl(
                throttle=float(msg.throttle), brake=float(msg.brake), steer=steer
            )
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CarlaBridge()
    try:
        node.setup()
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:  # noqa: BLE001 - log, then clean up the server below
        node.get_logger().error(f"bridge stopped: {type(e).__name__}: {e}")
        raise
    finally:
        node.teardown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
