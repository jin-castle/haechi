cmake_minimum_required(VERSION 3.8)
project(fusion_imu_radar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(haechi_cmake_compat REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ti_mmwave_rospkg_msgs REQUIRED)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
get_filename_component(_pcl_conversions_prefix "${pcl_conversions_DIR}/../../.." ABSOLUTE)
set(_pcl_conversions_include_dir "${_pcl_conversions_prefix}/include/pcl_conversions")
if(EXISTS "${_pcl_conversions_include_dir}/pcl_conversions")
  include_directories(${_pcl_conversions_include_dir})
endif()

# ------ 여기부터 추가 ------
# 실행 파일 생성
add_executable(fusion_imu_radar_node
  src/fusion_imu_radar_node.cpp
)

# include 경로 (PCL 헤더용)
target_include_directories(fusion_imu_radar_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

# 타겟에 의존성 연결
haechi_target_dependencies(fusion_imu_radar_node
  rclcpp
  sensor_msgs
  geometry_msgs
  ti_mmwave_rospkg_msgs
  pcl_conversions
)

# PCL 라이브러리 링크
target_link_libraries(fusion_imu_radar_node
  ${PCL_LIBRARIES}
)

# 설치 규칙 (ros2 run 할 수 있게)
install(TARGETS
  fusion_imu_radar_node
  DESTINATION lib/${PROJECT_NAME}
)
# ------ 여기까지 추가 ------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ament_package() 는 항상 제일 아래!
ament_package()

