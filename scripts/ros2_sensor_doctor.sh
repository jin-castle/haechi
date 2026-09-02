#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "${PRETTY_NAME:-unknown}"
else
  uname -a
fi

echo
echo "== ROS 2 =="
if ! command -v ros2 >/dev/null 2>&1; then
  for setup_file in /opt/ros/lyrical/setup.bash /opt/ros/jazzy/setup.bash; do
    if [[ -f "${setup_file}" ]]; then
      set +u
      source "${setup_file}"
      set -u
      break
    fi
  done
fi

if command -v ros2 >/dev/null 2>&1; then
  echo "ros2: $(command -v ros2)"
  echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
else
  echo "ros2 command not found. Run scripts/install_ros2_ubuntu.sh, then source the printed setup.bash path."
fi

echo
echo "== Build tools =="
for tool in python3 colcon rosdep; do
  if command -v "${tool}" >/dev/null 2>&1; then
    echo "${tool}: $(command -v "${tool}")"
  else
    echo "${tool}: missing"
  fi
done

echo
echo "== Sensor devices =="
shopt -s nullglob
serial_devices=(/dev/ttyACM* /dev/ttyUSB*)
video_devices=(/dev/video*)
if ((${#serial_devices[@]})); then
  printf 'serial: %s\n' "${serial_devices[@]}"
else
  echo "serial: none detected"
fi
if ((${#video_devices[@]})); then
  printf 'video: %s\n' "${video_devices[@]}"
else
  echo "video: none detected"
fi

echo
echo "== firesight_ros2 workspace =="
for setup_file in install/setup.bash ros2_ws/install/setup.bash; do
  if [[ -f "${setup_file}" ]]; then
    set +u
    source "${setup_file}"
    set -u
    break
  fi
done

if ! command -v ros2 >/dev/null 2>&1; then
  for setup_file in /opt/ros/lyrical/setup.bash /opt/ros/jazzy/setup.bash; do
    if [[ -f "${setup_file}" ]]; then
      set +u
      source "${setup_file}"
      set -u
      break
    fi
  done
fi

if command -v ros2 >/dev/null 2>&1; then
  if ros2 pkg prefix firesight_ros2 >/dev/null 2>&1; then
    echo "firesight_ros2: built and discoverable"
    ros2 pkg executables firesight_ros2 || true
  else
    echo "firesight_ros2: not discoverable yet"
    echo "Build from ros2_ws with: rosdep install --from-paths src --ignore-src -r -y && colcon build --symlink-install"
  fi
fi
