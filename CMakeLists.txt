cmake_minimum_required(VERSION 3.8)
project(visual_odometry)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DWITH_CUDA)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4.8 REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree search surface filters registration sample_consensus segmentation)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# ─── rtabmap core (vendored source, compiled as part of this package) ────────
set(RTABMAP_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rtabmap_core)

file(GLOB_RECURSE RTABMAP_CORE_SRCS  ${RTABMAP_CORE_DIR}/corelib_src/*.cpp)
file(GLOB_RECURSE RTABMAP_UTIL_SRCS  ${RTABMAP_CORE_DIR}/utilite_src/*.cpp)

add_library(rtabmap_vendored STATIC ${RTABMAP_CORE_SRCS} ${RTABMAP_UTIL_SRCS})

target_include_directories(rtabmap_vendored PUBLIC
  ${RTABMAP_CORE_DIR}/corelib_include
  ${RTABMAP_CORE_DIR}/utilite_include
  ${RTABMAP_CORE_DIR}/generated
  ${RTABMAP_CORE_DIR}/corelib_src
)
target_include_directories(rtabmap_vendored SYSTEM PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${ZED_INCLUDE_DIRS}
  /usr/local/zed/include
)
target_link_libraries(rtabmap_vendored
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  gtsam
  sqlite3 z
  ${ZED_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rtabmap_vendored OpenMP::OpenMP_CXX)
endif()
target_compile_options(rtabmap_vendored PRIVATE -w)

# ─── ROS message generation ─────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VOState.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ─── Main target ────────────────────────────────────────────────────────────
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  /usr/local/zed/include
  ${CMAKE_INSTALL_PREFIX}/include
  ${RTABMAP_CORE_DIR}/corelib_include
  ${RTABMAP_CORE_DIR}/utilite_include
  ${RTABMAP_CORE_DIR}/generated
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})

set(SOURCES
  src/main.cpp
  src/vo_node.cpp
  src/image_processor.cpp
  src/feature_detector.cpp
  src/feature_matcher.cpp
  src/zed_interface.cpp
  src/visualization.cpp
  src/resource_monitor.cpp
  src/logger.cpp
  src/vio_manager.cpp
)

add_executable(vo_node ${SOURCES})

target_link_libraries(vo_node
  rtabmap_vendored
  ${OpenCV_LIBRARIES}
  ${ZED_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${PCL_LIBRARIES}
  gtsam
  sqlite3 z octomap octomath
  "${cpp_typesupport_target}"
)

ament_target_dependencies(vo_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  geometry_msgs
  std_msgs
  tf2_ros
  tf2
)

install(TARGETS vo_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
ament_package()
