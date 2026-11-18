#!/usr/bin/env bash
# Build the ROS2 packages (M5) in a Linux-side workspace that links back to this repo.
# Runs in: WSL2 Ubuntu 22.04 (after sim/setup_wsl.sh), e.g.
#   bash /mnt/d/Project_ai/ros2_ws/build_wsl.sh
# Why not build in ros2_ws/ directly: build/ install/ log/ would land on the Windows drive
# (/mnt/d is slow through 9p, and symlinks there need Windows developer mode). The workspace
# ~/aiml04_ws/src holds symlinks to ros2_ws/src/<pkg>; with --symlink-install the installed
# Python nodes and launch file point back into the repo, so code edits need no rebuild
# (only .msg changes or new entry points do).
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
WS="${AIML04_WS:-$HOME/aiml04_ws}"
mkdir -p "$WS/src"
for p in perception_msgs perception_ros; do
  ln -sfn "$REPO/ros2_ws/src/$p" "$WS/src/$p"
done
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
cd "$WS"
colcon build --symlink-install --event-handlers console_direct-
echo
echo "Built in $WS. In every new WSL shell:"
echo "  source /opt/ros/humble/setup.bash && source $WS/install/setup.bash"
echo "  export AIML04_REPO_ROOT=$REPO RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1"
echo BUILD_WS_OK
