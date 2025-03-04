cmake_minimum_required(VERSION 3.8)
project(visual_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(vo_node src/vo_node.cpp)
ament_target_dependencies(vo_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

target_link_libraries(vo_node
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  vo_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()