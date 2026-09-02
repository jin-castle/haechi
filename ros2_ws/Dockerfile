FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools \
    ros-jazzy-rviz2 \
    ros-jazzy-tf2-ros \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ros2_ws
COPY src ./src

RUN source /opt/ros/jazzy/setup.bash \
  && rosdep update \
  && rosdep install --from-paths src --ignore-src -r -y \
  && colcon build --symlink-install

CMD ["/bin/bash"]
