#!/usr/bin/env bash
# M0.8: ROS2 Humble (ros-base) + CARLA 0.9.15 client in WSL2 Ubuntu 22.04.
# Runs in: 💻 WSL2 Ubuntu 22.04, as root:
#   wsl -d Ubuntu-22.04 -u root -- bash /mnt/d/Project_ai/sim/setup_wsl.sh
# Safe to re-run. No GUI tools (rviz etc.) are installed, to keep the laptop load low.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -q
apt-get install -y -q software-properties-common curl locales
locale-gen en_US en_US.UTF-8 >/dev/null
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
add-apt-repository -y universe >/dev/null

# ROS2 apt source (docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ver=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  curl -sSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${codename}_all.deb"
  dpkg -i /tmp/ros2-apt-source.deb
fi

apt-get update -q
apt-get install -y -q ros-humble-ros-base ros-humble-cv-bridge ros-humble-rmw-cyclonedds-cpp \
  python3-colcon-common-extensions python3-pip python3-numpy

grep -q "/opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
# With .wslconfig networkingMode=mirrored, `lo` is not multicast-capable and the default Fast DDS
# never discovers peers (nodes run but no message arrives; checked 2026-11-18). Cyclone DDS on
# loopback works. stack.launch.py sets the same defaults for its nodes.
grep -q "RMW_IMPLEMENTATION" ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=1
EOF

# CARLA client + light Python deps. numpy stays at Ubuntu's 1.x because cv_bridge is built
# against it; opencv is pinned to a release that still supports numpy 1.x.
python3 -m pip install -q --upgrade pip
python3 -m pip install -q "carla==0.9.15" "numpy<2" "opencv-python-headless<4.11" pyyaml

# Live perception runtime for perception_node (M5.3): ONNX Runtime + ByteTrack (supervision).
# No PyTorch. Skip with WITH_RUNTIME=0 (e.g. for the M0.8 smoke test only). TensorRT (M6.4)
# is installed separately by deploy/ instructions.
if [ "${WITH_RUNTIME:-1}" = "1" ]; then
  python3 -m pip install -q "numpy<2" "onnxruntime-gpu>=1.19,<1.23" "supervision>=0.22,<0.27"
fi

set +u  # ROS setup scripts read unset variables
source /opt/ros/humble/setup.bash
set -u
echo "ROS_DISTRO=$ROS_DISTRO"
python3 -c "import carla, cv2, numpy; print('carla client OK, cv2', cv2.__version__, 'numpy', numpy.__version__)"
echo SETUP_WSL_OK
