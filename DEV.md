# Visual Odometry 개발서

> **업데이트 규칙**: 개발/수정/버그픽스 시마다 이 문서를 반드시 업데이트할 것. 사소한 변경이라도 기록.
>
> **용도**: (1) 개발 시 빠른 참조·디버깅 (2) 논문화 시 초안 작성 기반

---

## 개발 시 빠른 참조

### 명령어

```bash
# 빌드 (workspace 루트에서)
cd ~/visual_odometry && colcon build --packages-select visual_odometry

# 실행
# ros2 모드: ZED wrapper 먼저 → ros2 launch visual_odometry vo.launch.py (input.source: ros2)
# zed_sdk 모드: vo.launch.py만 실행 (input.source: zed_sdk, ZED SDK 직접 연결)
ros2 launch visual_odometry vo.launch.py

# 토픽 확인
ros2 topic list | grep zed
ros2 topic echo /zed/zed_node/rgb/color/rect/image --no-arr
```

### 기능별 수정 위치

| 추가/수정할 기능 | 수정 파일 | 함수/위치 |
|------------------|-----------|-----------|
| 파이프라인 단계 추가 | `frame_processor.cpp` | `processFrame()` |
| 특징점 알고리즘 변경 | `feature_detector.cpp` | `detectFeatures()` |
| 매칭 로직 변경 | `feature_matcher.cpp` | `match()` |
| PnP/포즈 추정 | `frame_processor` 또는 새 `pose_estimator` | (추가 예정) |
| 파라미터 추가 | `vo_params.yaml` + `vo_node.cpp` | `declareParameters()`, `applyCurrentParameters()` |
| 시각화 추가 | `visualization.cpp` | `visualize()` |
| 메시지 발행 | `vo_node.cpp` | `processImages()` 내, `publishResults()` |

### 디버깅 체크리스트

- [ ] ZED 노드 먼저 실행됐는지
- [ ] `ros2 topic list`로 토픽 퍼블리시 확인
- [ ] `Camera parameters received` 로그 나오는지 (camera_info 수신)
- [ ] `vo_params.yaml` 토픽 경로가 ZED wrapper와 일치하는지

### 개발 시 자주 쓰는 수식 (복붙용)

```
# 2D + depth → 3D backproject
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth.at<float>(v, u)

# solvePnP 입력
objectPoints: prev 프레임 3D (Nx3)
imagePoints: curr 프레임 2D (Nx2)
cameraMatrix: K from camera_params_
```

---

## 논문 매핑 (논문화 시 참조)

| 논문 섹션 | DEV.md 참조 | 비고 |
|-----------|--------------|------|
| **Abstract** | §1, §2 | 목적, 환경, 파이프라인 요약 |
| **1. Introduction** | §1, §2 | 배경, 연구 목적, 기여 |
| **2. Related Work** | (별도 작성) | VO/SLAM 선행연구 |
| **3. System Overview** | §3, §3.3 | 전체 구조, 데이터 흐름 |
| **4. Method** | §2, §3.2, §4 | 파이프라인, 알고리즘(특징점검출/매칭/예정:PnP) |
| **5. Implementation** | §3.1, §5, §6 | 코드 구조, 파라미터, ROS2 설정 |
| **6. Experiments** | §8, §9 | 버그/이슈, 실험 설정, 결과(추후 추가) |
| **7. Conclusion** | §2, §11 | 완료/미완료, 향후 계획 |

---

## 1. 프로젝트 개요

- **목적**: RGB-D 카메라(ZED) 기반 Visual Odometry
- **환경**: ROS2 Humble, OpenCV 4.8, ZED SDK 3.x, CUDA
- **플랫폼**: aarch64 (Jetson 등)

---

## 2. 파이프라인 단계

| # | 단계 | 상태 | 비고 |
|---|------|------|------|
| 1 | 이미지 획득 | ✅ 완료 | RGB + Depth |
| 2 | 특징점 검출 | ✅ 완료 | ORB |
| 3 | 특징점 매칭 | ✅ 완료 | 2D-2D, RANSAC |
| 4 | 카메라 내부 파라미터 | ✅ 완료 | camera_info |
| 5 | RGB-D 동기화 | ✅ 완료 | ros2: current_depth_ 전달, zed_sdk: 큐 경로 |
| 6 | 3D 점 생성 | ❌ 미완료 | depth backproject |
| 7 | PnP 포즈 추정 | ❌ 미완료 | solvePnPRansac |
| 8 | 포즈 누적 | ❌ 미완료 | T_global 누적 |
| 9 | 결과 발행 | ❌ 미완료 | pose_pub_, vo_state_pub_ |
| 10 | (선택) 스케일/최적화 | ❌ 미완료 | - |

---

## 3. 코드 구조

### 3.1 디렉터리

```
visual_odometry/
├── config/
│   └── vo_params.yaml      # 파라미터
├── include/visual_odometry/
│   ├── feature_detector.hpp
│   ├── feature_matcher.hpp
│   ├── frame_processor.hpp
│   ├── image_processor.hpp
│   ├── logger.hpp
│   ├── resource_monitor.hpp
│   ├── types.hpp
│   ├── visualization.hpp
│   ├── vo_node.hpp
│   └── zed_interface.hpp
├── launch/
│   └── vo.launch.py
├── msg/
│   └── VOState.msg
├── src/
│   ├── feature_detector.cpp
│   ├── feature_matcher.cpp
│   ├── frame_processor.cpp
│   ├── image_processor.cpp
│   ├── logger.cpp
│   ├── main.cpp
│   ├── resource_monitor.cpp
│   ├── visualization.cpp
│   ├── vo_node.cpp
│   └── zed_interface.cpp
└── DEV.md                  # 이 문서
```

### 3.2 주요 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| `vo_node` | ROS2 노드, 콜백/구독/발행, 처리 루프 |
| `FrameProcessor` | preprocess → detect → match 파이프라인 |
| `FeatureDetector` | FAST + ORB, max_features/scale_factor/n_levels |
| `FeatureMatcher` | BFMatcher(knnMatch) + Lowe ratio + findFundamentalMat(RANSAC) |
| `Visualizer` | original/features/matches 윈도우 |
| `Logger` | 시스템 정보, 메트릭, 파라미터 변경 로그 |
| `ZEDInterface` | ZED SDK 직접 연결 (input_source=zed_sdk) |

### 3.3 데이터 흐름

**ros2 모드**:
```
rgb_sub_  ──► rgbCallback ──► depth_mutex_로 current_depth_ 복사 ──► processImages(rgb, depth)
depth_sub_ ─► depthCallback ─► depth_mutex_로 current_depth_ 갱신
camera_info_sub_ ─► cameraInfoCallback ─► camera_params_
```

**zed_sdk 모드**:
```
zed_timer_ ──► getImages(rgb,depth) ──► image_queue_ ──► processingLoop ──► processImages(rgb,depth)
rgbCallback, depthCallback: early return (무시)
```

**공통**: FrameProcessor::processFrame → preprocessFrame → detectFeatures → matchFeatures

---

## 4. 타입 정의 (types.hpp)

| 구조체 | 용도 |
|--------|------|
| `CameraParams` | fx, fy, cx, cy, width, height, getCameraMatrix() |
| `Features` | keypoints, descriptors, visualization |
| `FeatureMatches` | matches, prev_points, curr_points (2D) |
| `ProcessingResult` | features, matches, feature_detection_time, feature_matching_time |

---

## 5. 파라미터 (vo_params.yaml)

### 5.1 입력

| 경로 | 기본값 | 설명 |
|------|--------|------|
| `input.source` | "zed_sdk" | "ros2" \| "zed_sdk" |
| `topics.rgb_image` | /zed/zed_node/rgb/color/rect/image | ZED 버전에 따라 다를 수 있음 |
| `topics.depth_image` | /zed/zed_node/depth/depth_registered | |
| `topics.camera_info` | /zed/zed_node/rgb/color/rect/camera_info | |

**참고**: 일부 ZED wrapper는 `/zed/zed_node/rgb/image_rect_color`, `/zed/zed_node/rgb/camera_info` 사용.

### 5.2 처리

| 경로 | 기본값 |
|------|--------|
| `processing.enable_pose_estimation` | false |
| `processing.publish_results` | true |

### 5.3 특징점

| 경로 | 기본값 |
|------|--------|
| `feature_detector.max_features` | 1000 |
| `feature_detector.scale_factor` | 1.2 |
| `feature_detector.n_levels` | 8 |
| `feature_detector.matching.ratio_threshold` | 0.8 |
| `feature_detector.matching.ransac.threshold` | 3.0 |

---

## 6. QoS 설정

| 구독 | QoS |
|------|-----|
| rgb_image | `QoS(1).best_effort().durability_volatile()` |
| depth_image | `SensorDataQoS()` |
| camera_info | `SensorDataQoS()` |

ZED sensor 토픽과 호환되도록 설정됨.

---

## 7. 발행 토픽 (선언만, 미구현)

| 토픽 | 타입 | 용도 |
|------|------|------|
| camera_pose | geometry_msgs/PoseStamped | 카메라 포즈 |
| vo_state | visual_odometry/VOState | 포즈 + 특징점 수 + 품질 메트릭 |

`publishResults()` 선언만 있고 구현 없음.

---

## 8. 버그 목록

| # | 버그 | 영향 | 수정 방향 |
|---|------|------|-----------|
| 1 | ~~**Source 로그 오류**~~ | ~~yaml 값 미반영~~ | ✅ 수정: `get_parameter("input.source")` 직접 읽기 |
| 2 | ~~**zed_sdk 모드 camera_params_ 미설정**~~ | ~~camera_info 토픽 없음~~ | ✅ 수정: connect 성공 후 `getCameraParameters()` + `getResolution()` 호출 |
| 3 | **prev_depth_ 의미** | `depthCallback`에서 `current_depth_.copyTo(prev_depth_)` → 매 프레임 동일. "이전 프레임" depth가 아님. | PnP에서 prev 3D 필요 시, rgb 프레임 단위로 prev_depth 저장하도록 로직 수정 |

---

## 9. 알려진 이슈 (버그 아님)

1. **ros2 RGB-D 근사 동기화**: rgb/depth 별도 콜백이라 타임스탬프 정확 동기화 아님. 최신 depth 사용.
2. **camera_info 초기 로그**: 수신 전에는 "N/A (waiting for camera_info...)" 출력 (정상).
3. **Gtk-Message**: `Failed to load module "canberra-gtk-module"` — 무시 가능.
4. **OpenCV 버전 충돌**: cv_bridge(4.5d) vs OpenCV 4.8 링커 경고 — 동작에는 영향 없음.

---

## 10. 변경 이력 (Changelog)

### 2026-03-10

- **버그 수정 #1, #2**: Source 로그 → get_parameter 직접 읽기. zed_sdk camera_params_ → connect 후 getCameraParameters()+getResolution() 호출.
- **RGB-D 동기화 (ros2/zed_sdk)**: ros2 모드에서 rgbCallback이 current_depth_를 processImages에 전달. depth_mutex_로 동시 접근 보호. zed_sdk 모드에서는 rgb/depth 콜백 early return, 큐 경로만 사용.
- **DEV.md 생성**: 개발서 초안 작성
- **camera_params 초기화**: `logSystemInfo` 시 camera_width/height/fx/fy/cx/cy 미초기화로 쓰레기 값 출력 → 0으로 초기화, N/A 메시지 출력으로 수정
- **QoS 변경**: camera_info, depth 구독을 `SensorDataQoS()`로 변경 (ZED 호환)
- **camera_info 로그**: 수신 시 `Camera parameters received: WxH, fx=... fy=...` 출력 추가

---

## 11. 다음 작업 체크리스트

- [x] rgbCallback에서 current_depth_를 processImages에 전달 (ros2/zed_sdk 모드 분리)
- [ ] 3D 점 생성: prev_points 2D + depth → backproject
- [ ] PnP: solvePnPRansac(prev_3d, curr_2d, K) → R, t
- [ ] 포즈 누적: T_global
- [ ] publishResults 구현: pose_pub_, vo_state_pub_

---

## 12. 논문용 메모 (실험/결과 추후 기록)

- **실험 환경**: (하드웨어, OS, ROS 버전 등)
- **데이터셋**: (사용한 시퀀스, 경로)
- **평가 지표**: (ATE, RPE, 처리 속도 등)
- **결과**: (표, 그래프용 수치)
