cmake_minimum_required(VERSION 3.8)
project(visual_odometry)

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

# 헤더 파일 포함 디렉토리 설정
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# 소스 파일 목록
set(SOURCES
  src/main.cpp
  src/vo_node.cpp
  src/image_processor.cpp
  src/feature_detector.cpp
)

# 실행 파일 추가
add_executable(vo_node ${SOURCES})

# 의존성 링크
target_link_libraries(vo_node
  ${OpenCV_LIBRARIES}
)

# ament 의존성 추가
ament_target_dependencies(vo_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
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

# 패키지 설정 파일 생성
ament_package()

# 링커 옵션 추가
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")