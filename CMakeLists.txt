cmake_minimum_required(VERSION 3.8)
project(visual_odometry)

# CUDA 설정
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DWITH_CUDA)
endif()

# 컴파일러 설정
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 4.8 명시적 지정
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4.8 REQUIRED)

# 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 메시지 생성 (ament_package 전에 호출)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VOState.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# 실행 파일이 메시지 인터페이스를 사용할 수 있도록 설정
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 헤더 파일 포함 디렉토리 설정
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  /usr/local/zed/include      # ZED SDK 헤더 경로 추가
  ${CMAKE_INSTALL_PREFIX}/include  # 생성된 메시지 헤더 경로
)

# 소스 파일 목록
set(SOURCES
  src/main.cpp
  src/vo_node.cpp
  src/image_processor.cpp
  src/feature_detector.cpp
  src/feature_matcher.cpp
  src/zed_interface.cpp
  src/visualization.cpp
  src/frame_processor.cpp
)

# 실행 파일 추가
add_executable(vo_node ${SOURCES})

# 의존성 링크
target_link_libraries(vo_node
  ${OpenCV_LIBRARIES}
  ${ZED_LIBRARIES}
  ${CUDA_LIBRARIES}
  "${cpp_typesupport_target}"
)

# ament 의존성 추가
ament_target_dependencies(vo_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  geometry_msgs
  std_msgs
)

# 설치 규칙
install(TARGETS vo_node
  DESTINATION lib/${PROJECT_NAME}
)

# 헤더 파일 설치
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# 링커 옵션 추가
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")

# 패키지 설정 파일 생성 (마지막에 호출)
ament_package()