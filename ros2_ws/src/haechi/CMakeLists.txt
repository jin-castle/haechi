cmake_minimum_required(VERSION 3.8)
project(haechi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(haechi_cmake_compat REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# add_executable(haechi_node src/haechi_node.cpp)
# ament_target_dependencies(haechi_node rclcpp sensor_msgs geometry_msgs)

# Localizer
add_executable(haechi_localizer_node
  src/haechi_localizer_node.cpp)
haechi_target_dependencies(haechi_localizer_node
  rclcpp sensor_msgs geometry_msgs)

# Map Matcher
add_executable(haechi_map_matcher_node
  src/haechi_map_matcher_node.cpp)
haechi_target_dependencies(haechi_map_matcher_node
  rclcpp sensor_msgs geometry_msgs)

install(TARGETS 
  # haechi_node
  haechi_localizer_node
  haechi_map_matcher_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
