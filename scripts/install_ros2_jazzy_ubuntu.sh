#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-jazzy}"
exec bash "${SCRIPT_DIR}/install_ros2_ubuntu.sh"
