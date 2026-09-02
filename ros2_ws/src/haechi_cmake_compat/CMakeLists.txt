cmake_minimum_required(VERSION 3.8)
project(haechi_cmake_compat)

find_package(ament_cmake REQUIRED)

install(DIRECTORY cmake/
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_package(CONFIG_EXTRAS cmake/HaechiLyricalCompat.cmake)
