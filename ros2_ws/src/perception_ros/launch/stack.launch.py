"""Full closed loop (M5.7): bridge + perception + control (+ recorder, scenario, rosbag).

    # WSL2, CARLA already running on Windows
    ros2 launch perception_ros stack.launch.py carla_host:=localhost
    ros2 launch perception_ros stack.launch.py scenario:=stopped_car town:=Town01 load_town:=true

Arguments:
  carla_host / carla_port   CARLA server ('' / 0 = configs/carla.yaml, default localhost:2000)
  town, load_town           map (Town01 / Town02) and whether to load it if another is active
  config                    configs/carla.yaml (relative paths resolve from the repo root)
  scenario                  '' (free drive) | stopped_car | pedestrian_crossing | cut_in |
                            night_rain. The launch shuts down when the scenario script exits.
  record                    true: recorder_node writes media/<scenario or stack>_<time>.mp4
  record_bag                true: ros2 bag record of recorder.bag_topics into runs/bags/
  repo_root                 the AIML-04 checkout (default: $AIML04_REPO_ROOT or auto-detected)
"""

from __future__ import annotations

import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _setup(context, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003 - launch API
    import yaml
    from perception_ros.repo import ENV, repo_root

    lc = {k: LaunchConfiguration(k).perform(context) for k in (
        "carla_host", "carla_port", "town", "load_town", "config", "scenario", "record",
        "record_bag", "repo_root",
    )}
    root = lc["repo_root"] or str(repo_root())
    env = {
        ENV: root,
        "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p),
        # WSL2 mirrored networking: Fast DDS finds no peers on the non-multicast `lo`, Cyclone
        # DDS on loopback works (docs/SETUP.md 6). An RMW set in the shell wins.
        "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        "ROS_LOCALHOST_ONLY": os.environ.get("ROS_LOCALHOST_ONLY", "1"),
    }
    config = lc["config"]
    cfg_path = config if os.path.isabs(config) else os.path.join(root, config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = {"config": cfg_path}
    scenario = lc["scenario"].strip()
    host = lc["carla_host"] or cfg["carla"]["host"]
    port = int(lc["carla_port"] or cfg["carla"]["port"])

    actions = [
        LogInfo(msg=f"[stack] repo {root}, CARLA {host}:{port}, scenario {scenario or '-'}"),
        Node(
            package="perception_ros", executable="carla_bridge_node", name="carla_bridge_node",
            output="screen", cwd=root, additional_env=env,
            parameters=[{**common, "carla_host": host, "carla_port": port,
                         "town": lc["town"], "load_town": _truthy(lc["load_town"])}],
        ),
        Node(
            package="perception_ros", executable="perception_node", name="perception_node",
            output="screen", cwd=root, additional_env=env, parameters=[common],
        ),
        Node(
            package="perception_ros", executable="control_node", name="control_node",
            output="screen", cwd=root, additional_env=env, parameters=[common],
        ),
    ]
    if _truthy(lc["record"]):
        actions.append(Node(
            package="perception_ros", executable="recorder_node", name="recorder_node",
            output="screen", cwd=root, additional_env=env,
            parameters=[{**common, "name": scenario or "stack"}],
        ))
    if _truthy(lc["record_bag"]):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bag = os.path.join(root, "runs", "bags", f"{scenario or 'stack'}_{stamp}")
        actions.append(ExecuteProcess(
            cmd=["ros2", "bag", "record", "-o", bag, *cfg["recorder"]["bag_topics"]],
            additional_env=env, output="screen",
        ))
    if scenario:
        proc = ExecuteProcess(
            cmd=["python3", "-m", "sim.scenarios.run", "--scenario", scenario, "--host", host,
                 "--port", str(port), "--config", cfg_path],
            cwd=root, additional_env=env, output="screen", name=f"scenario_{scenario}",
        )
        actions += [
            proc,
            RegisterEventHandler(OnProcessExit(
                target_action=proc,
                on_exit=[EmitEvent(event=Shutdown(reason=f"scenario {scenario} finished"))],
            )),
        ]
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("carla_host", default_value="localhost"),
        DeclareLaunchArgument("carla_port", default_value=""),
        DeclareLaunchArgument("town", default_value=""),
        DeclareLaunchArgument("load_town", default_value="false"),
        DeclareLaunchArgument("config", default_value="configs/carla.yaml"),
        DeclareLaunchArgument("scenario", default_value=""),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("record_bag", default_value="false"),
        DeclareLaunchArgument("repo_root", default_value=""),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=_setup)])
