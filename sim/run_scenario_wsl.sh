#!/usr/bin/env bash
# Runs in: WSL2 Ubuntu 22.04 (root). One closed-loop scenario with recording; log -> runs/stack_<scenario>.log
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u root -- bash /mnt/d/Project_ai/sim/run_scenario_wsl.sh night_rain
# Needs CARLA already listening on localhost:2000 (sim/restart_carla.ps1) and ros2_ws built (ros2_ws/build_wsl.sh).
set +u
source /opt/ros/humble/setup.bash
source /root/aiml04_ws/install/setup.bash
export AIML04_REPO_ROOT=/mnt/d/Project_ai RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1
cd /mnt/d/Project_ai
timeout 480 ros2 launch perception_ros stack.launch.py town:=Town01 load_town:=true scenario:=${1:-stopped_car} > runs/stack_${1:-stopped_car}.log 2>&1
echo exit=$?
