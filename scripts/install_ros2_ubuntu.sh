#!/usr/bin/env bash
set -euo pipefail

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required for ROS 2 apt installation." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot detect Ubuntu release: /etc/os-release is missing." >&2
  exit 1
fi

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
TARGET_USER="${ROS_SETUP_USER:-${SUDO_USER:-$(id -un)}}"
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"

if [[ -z "${TARGET_HOME}" ]]; then
  echo "Cannot resolve home directory for ${TARGET_USER}." >&2
  exit 1
fi

if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "Expected Ubuntu for ROS 2 apt installation; detected ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

if [[ -z "${ROS_DISTRO_NAME:-}" ]]; then
  case "${CODENAME}" in
    noble)
      ROS_DISTRO_NAME="jazzy"
      ;;
    resolute)
      ROS_DISTRO_NAME="lyrical"
      ;;
    *)
      echo "Unsupported Ubuntu codename for this setup: ${CODENAME:-unknown}." >&2
      echo "Use Ubuntu 24.04 Noble, Ubuntu 26.04 Resolute, or the ros2_ws Dockerfile." >&2
      exit 1
      ;;
  esac
fi

case "${ROS_DISTRO_NAME}:${CODENAME}" in
  jazzy:noble | lyrical:resolute)
    ;;
  *)
    echo "ROS_DISTRO_NAME=${ROS_DISTRO_NAME} does not match Ubuntu ${CODENAME}." >&2
    echo "Supported pairs: jazzy+noble, lyrical+resolute." >&2
    exit 1
    ;;
esac

sudo apt update
sudo apt install -y \
  curl \
  gnupg \
  lsb-release \
  locales \
  software-properties-common

sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo add-apt-repository universe -y
sudo apt update

ROS_APT_SOURCE_VERSION="$(
  curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F '"tag_name"' \
    | awk -F'"' '{print $4}'
)"

curl -fsSL -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt install -y \
  "ros-${ROS_DISTRO_NAME}-desktop" \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  ros-dev-tools

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi

sudo -u "${TARGET_USER}" env HOME="${TARGET_HOME}" rosdep update

TARGET_BASHRC="${TARGET_HOME}/.bashrc"
if ! grep -F "source /opt/ros/${ROS_DISTRO_NAME}/setup.bash" "${TARGET_BASHRC}" >/dev/null 2>&1; then
  printf '\nsource /opt/ros/%s/setup.bash\n' "${ROS_DISTRO_NAME}" | sudo tee -a "${TARGET_BASHRC}" >/dev/null
  sudo chown "${TARGET_USER}:${TARGET_USER}" "${TARGET_BASHRC}"
fi

echo "ROS 2 ${ROS_DISTRO_NAME} is installed for Ubuntu ${CODENAME}."
echo "Open a new shell or run: source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
