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

# RViz에서 TF 시각화
# 1. ros2 launch visual_odometry vo.launch.py 실행
# 2. rviz2 실행 → Fixed Frame: odom
# 3. Add → TF → odom, camera_link 축 표시
```

### 기능별 수정 위치

| 추가/수정할 기능 | 수정 파일 | 함수/위치 |
|------------------|-----------|-----------|
| 파이프라인 단계 추가 | `frame_processor.cpp` | `processFrame()` |
| 특징점 알고리즘 변경 | `feature_detector.cpp` | `detectFeatures()` |
| 매칭 로직 변경 | `feature_matcher.cpp` | `match()` |
| PnP/포즈 추정 | `frame_processor.cpp` | `processFrame()` 내 solvePnPRansac |
| 파라미터 추가 | `vo_params.yaml` + `vo_node.cpp` | `declareParameters()`, `applyCurrentParameters()` |
| 시각화 추가 | `visualization.cpp` | `visualize()` |
| 메시지 발행 | `vo_node.cpp` | `processImages()` 내, `publishResults()` |
| IMU-VO fusion | `imu_fusion.hpp`, `vo_node.cpp` | fusion_mode, complementary/ekf/factor_graph |
| Factor graph | `factor_graph.hpp`, `factor_graph.cpp`, `vo_node.cpp` | §11 Phase 2~5 |

### 디버깅 체크리스트

- [ ] zed_sdk: ZED USB 연결됐는지 / ros2: ZED wrapper 먼저 실행
- [ ] `ros2 topic list`로 토픽 퍼블리시 확인 (ros2 모드)
- [ ] `Camera parameters received` 또는 `Camera parameters (ZED SDK)` 로그
- [ ] `vo_params.yaml` 토픽 경로가 ZED wrapper와 일치하는지 (ros2)
- [ ] `3D Points: N` > 0 (Performance Metrics)

### 개발 시 자주 쓰는 수식 (복붙용)

```
# 2D + depth → 3D backproject (ZED depth: mm)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth.at<float>(v, u)

# solvePnP 입력 (curr 3D + prev 2D)
objectPoints: prev_points_3d (curr 프레임 3D)
imagePoints: prev_points (prev 프레임 2D)
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
| 6 | 3D 점 생성 | ✅ 완료 | curr_points + curr_depth → backprojectAndFilter (ZED mm) |
| 7 | PnP 포즈 추정 | ✅ 완료 | solvePnPRansac, R,t → 포즈 누적 |
| 8 | 포즈 누적 | ✅ 완료 | T_global 누적, Pose (x,y,z) m 로그 |
| 9 | 결과 발행 | ✅ 완료 | camera_pose, vo_state (publish_results 파라미터) |
| 9a | IMU 발행 | ✅ 완료 | zed_sdk 모드, ZED2/ZED Mini, sensor_msgs/Imu |
| 9b | IMU-VO fusion | ✅ 완료 | complementary, EKF 15-state (p,v,rpy,bias), factor_graph stub |
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
│   ├── factor_graph.hpp       # (예정) FactorGraphBackend
│   ├── image_processor.hpp
│   ├── logger.hpp
│   ├── imu_fusion.hpp
│   ├── imu_fusion_ekf.hpp
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
│   ├── factor_graph.cpp       # (예정)
│   ├── image_processor.cpp
│   ├── imu_fusion_ekf.cpp
│   ├── imu_fusion_factory.cpp
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
| `FrameProcessor` | preprocess → detect → match → backprojectAndFilter |
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
                    │
                    └── getSensorsData() ──► imu_pub_ (imu.enable, ZED2/ZED Mini만)
rgbCallback, depthCallback: early return (무시)
```

**공통**: processFrame → preprocess → detect → match → backprojectAndFilter(curr_points+depth)

---

## 4. 타입 정의 (types.hpp)

| 구조체 | 용도 |
|--------|------|
| `CameraParams` | fx, fy, cx, cy, width, height, getCameraMatrix() |
| `Features` | keypoints, descriptors, visualization |
| `FeatureMatches` | matches, prev_points, curr_points (2D), prev_points_3d (curr 3D, PnP용) |
| `ProcessingResult` | features, matches, feature_detection_time, feature_matching_time, visualization_time, is_keyframe |

---

## 5. 파라미터 (vo_params.yaml)

### 5.1 입력

| 경로 | 기본값 | 설명 |
|------|--------|------|
| `input.source` | "zed_sdk" | "ros2" \| "zed_sdk" |
| `topics.rgb_image` | /zed/zed_node/rgb/color/rect/image | ZED 버전에 따라 다를 수 있음 |
| `topics.depth_image` | /zed/zed_node/depth/depth_registered | |
| `topics.camera_info` | /zed/zed_node/rgb/color/rect/camera_info | |
| `topics.imu` | imu | IMU 발행 토픽 (zed_sdk, ZED2/ZED Mini) |
| `imu.enable` | true | IMU 발행 활성화 |
| `imu.fusion_mode` | "none" | "none" \| "complementary" \| "ekf" \| "factor_graph" |
| `imu.complementary_alpha` | 0.98 | complementary: gyro trust (0.95~0.99) |
| `topics.imu_sub` | /zed/zed_node/imu/data | IMU 구독 (ros2, fusion 시) |

**참고**: 일부 ZED wrapper는 `/zed/zed_node/rgb/image_rect_color`, `/zed/zed_node/rgb/camera_info` 사용.

### 5.2 처리

| 경로 | 기본값 |
|------|--------|
| `processing.enable_pose_estimation` | false | false 시 PnP/포즈 누적/발행 스킵 (odometry off) |
| `processing.publish_results` | true |

### 5.2a Frame IDs / TF

| 경로 | 기본값 | 설명 |
|------|--------|------|
| `frames.frame_id` | "odom" | PoseStamped, VOState, TF parent |
| `frames.child_frame_id` | "camera_link" | TF child (RViz) |
| `tf.publish` | true | TF 발행 여부 |

### 5.2b VO (zero_motion)

| 경로 | 기본값 | 설명 |
|------|--------|------|
| `vo.zero_motion_threshold_mm` | 2.0 | \|t\| < 이 값 |
| `vo.zero_motion_rotation_threshold_rad` | 0.002 | \|θ\| < 이 값. t와 θ 둘 다 작으면 포즈 누적 스킵 |

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

## 7. 발행 토픽

| 토픽 | 타입 | 용도 |
|------|------|------|
| camera_pose | geometry_msgs/PoseStamped | 카메라 포즈 |
| vo_state | visual_odometry/VOState | 포즈 + 특징점 수 + 품질 메트릭 |
| imu | sensor_msgs/Imu | IMU (zed_sdk, ZED2/ZED Mini, imu.enable) |

`publishResults()`: camera_pose (PoseStamped), vo_state (VOState) 발행. processing.publish_results로 on/off. TF: frames.frame_id→frames.child_frame_id. frames.*, tf.publish는 vo_params.yaml에서 설정.

---

## 8. 버그 목록

| # | 버그 | 영향 | 수정 방향 |
|---|------|------|-----------|
| 1 | ~~**Source 로그 오류**~~ | ~~yaml 값 미반영~~ | ✅ 수정: `get_parameter("input.source")` 직접 읽기 |
| 2 | ~~**zed_sdk 모드 camera_params_ 미설정**~~ | ~~camera_info 토픽 없음~~ | ✅ 수정: connect 성공 후 `getCameraParameters()` + `getResolution()` 호출 |
| 3 | ~~**prev_depth_ 의미**~~ | ~~prev_depth가 "이전 프레임" 아님~~ | ✅ 우회: curr_depth+curr_points 사용 (prev_depth NaN/미사용) |

---

## 9. 알려진 이슈 (버그 아님)

1. **ros2 RGB-D 근사 동기화**: rgb/depth 별도 콜백이라 타임스탬프 정확 동기화 아님. 최신 depth 사용.
2. **ZED depth 단위**: zed_interface에서 mm 반환 (7948=7.9m). backprojectAndFilter에서 50~20000 범위 사용.
3. **camera_info 초기 로그**: 수신 전에는 "N/A (waiting for camera_info...)" 출력 (정상).
4. **Gtk-Message**: `Failed to load module "canberra-gtk-module"` — 무시 가능.
5. **정지 시 드리프트**: depth/매칭 노이즈로 프레임당 소량 오차 누적. `vo.zero_motion_threshold_mm` (2mm) + `vo.zero_motion_rotation_threshold_rad` (0.002): |t|와 |θ| 둘 다 작으면 포즈 누적 스킵.
5a. **좌표계**: odom→camera_link (body, X fwd Y left Z up) + camera_link→camera_optical_frame (static). R_opt_to_body로 optical→body 변환.
5b. **동적 물체**: 카메라 정지 시 멀리 움직이는 물체가 있으면 VO가 잘못된 motion 추정. complementary는 yaw/pos를 VO에 맡겨 검증 없음.
6. **OpenCV 버전 충돌**: cv_bridge(4.5d) vs OpenCV 4.8 링커 경고 — 동작에는 영향 없음.

### 9a. 좌표계 (odom→camera_link→camera_optical_frame)

**TF 구조**: odom → camera_link (body) → camera_optical_frame (static)

| frame | convention |
|-------|------------|
| odom | body: X forward, Y left, Z up (REP 103) |
| camera_link | body: odom과 동일 축 정렬 (정지 시 I) |
| camera_optical_frame | optical: X right, Y down, Z forward (OpenCV) |

**R_opt_to_body** (optical→body): `[[0,0,1],[-1,0,0],[0,-1,0]]` — X_body=Z_opt, Y_body=-X_opt, Z_body=-Y_opt

**포즈 누적 (ZED wrapper 동일)**:
- solvePnP(curr_3d, prev_2d) → R,t = T_prev_from_curr (world→camera)
- 역변환 없이 직접 사용: `T_global_ = T_global_ * T_cp`
- T_global_ = T_0_from_curr: translation = curr 원점의 frame0 좌표 (mm→m /1000)

**위치**: `t_body = R_opt_to_body * t_curr_in_0`

**회전 (3단계)**:
1. R_body = R_opt_to_body * R_0_from_curr.t() * R_opt_to_body.t() — optical→body similarity 변환, 정지 시 I
2. TF 규약: odom→camera_link는 child→parent. p_odom = R_odom_from_camera_link * p_camera_link
3. 발행: R_tf = R_body.t() (R_odom_from_camera_link)

---

## 10. 변경 이력 (Changelog)

### 2026-03-20 — Phase 3: CombinedImuFactor 통합 (VIO)

- **CombinedImuFactor**: GTSAM의 IMU preintegration을 factor graph에 실제 적용
  - 심볼 규약: `x(i)` Pose3, `v(i)` Vector3 (velocity), `b(i)` imuBias::ConstantBias
  - `addVelocityBiasPrior(0)`: 첫 프레임에서 velocity=0, bias=0 prior 추가
  - `addImuFactor(i, j)`: CombinedImuFactor(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, PIM)
- **velocity 초기값**: NavState predict로 PIM에서 예측. bias는 이전 최적화 결과 전파.
- **optimize() 후**: 최적화된 velocity/bias를 `prev_velocity`, `prev_bias`에 저장 → 다음 프레임 초기값
- **sliding window 확장**: v/b 노드도 shift (k→k-1). 새 앵커(node 0)에 v/b prior 재설정. IMU factor는 slide 시 drop → 새 프레임부터 자동 추가.
- **흐름**: `fuse(vector)` → preintegrateImu → imu_preintegrated=true → `fuse()` → addPose + addVelocityBiasPrior(첫프레임) + addOdometryFactor + addImuFactor → optimize
- **수정 파일**: `factor_graph.hpp`, `factor_graph.cpp`, `imu_fusion_factor_graph.cpp`

### 2026-03-24 — Phase 2: GTSAM IMU Preintegration 설정

- **PreintegrationCombinedParams**: ZED 2i BMI088 IMU 기본 노이즈 파라미터 설정
  - `accel_noise_sigma=0.05 m/s²/√Hz`, `gyro_noise_sigma=0.005 rad/s/√Hz`
  - `accel_bias_rw=0.001`, `gyro_bias_rw=0.0001`, gravity=9.81 (Z-up)
- **PreintegratedCombinedMeasurements**: 프레임 간 IMU 샘플 → `integrateMeasurement(acc, gyro, dt)` 호출
- **dt 계산**: 연속 IMU 샘플 timestamp 차이 사용. 비정상(≤0 또는 >100ms) 시 5ms fallback
- **검증 로그**: `IMU preint: dt=... dp=(...) dv=(...) rpy=(...)deg` 매 프레임 출력
- **Phase 2 범위**: preintegration 계산만 수행, factor graph에 IMU factor 추가는 Phase 3
- **ImuFusionFactorGraph**: vector fuse() 오버로드에서 preintegrateImu() + logPreintegration() 호출
- **수정 파일**: `factor_graph.hpp`, `factor_graph.cpp`, `imu_fusion.hpp`, `imu_fusion_factor_graph.cpp`

### 2026-03-24 — IMU 고빈도 폴링 타이머 (200Hz)

- **문제**: ZED SDK `getSensorsData(TIME_REFERENCE::IMAGE)`는 이미지 동기화 IMU 1개만 반환 → 프레임당 0~2개 샘플 (preintegration에 부족)
- **해결**: `imuPollLoop()` 전용 스레드 200Hz(5ms sleep), `getSensorsData(TIME_REFERENCE::CURRENT)` 사용
- **ROS2 타이머 → 스레드 변경 이유**: SingleThreadedExecutor에서 200Hz 타이머가 zed_timer_(60Hz)와 경쟁하여 실질 0~3개만 획득. 전용 스레드로 executor 경합 해소.
- **ZED SDK thread-safe**: [Stereolabs 공식](https://github.com/stereolabs/zed-sdk/issues/89) — `grab()`과 `getSensorsData(CURRENT)` 병렬 호출 안전
- **레퍼런스**: VINS-Mono/ORB-SLAM3/Kimera-VIO 모두 고빈도 IMU 독립 수집 → 프레임 간 버퍼 누적 → drain하여 preintegration. 우리 구조도 동일 패턴.
- **getImages()에서 IMU 제거**: 이미지 획득과 IMU 획득 완전 분리.
- **ZEDInterface 확장**: `getSensorsDataCurrent()` 메서드 추가 (`TIME_REFERENCE::CURRENT` 래핑)
- **예상 결과**: `IMU buffer: N samples drained` N ≈ 5~7 @200Hz/~33fps VO
- **수정 파일**: `zed_interface.hpp`, `zed_interface.cpp`, `vo_node.hpp`, `vo_node.cpp`

### 2026-03-20 — IMU Preintegration Phase 1: IMU 버퍼링 인프라

- **IMU 버퍼 추가**: `latest_imu_`(단일 샘플) 외에 `std::deque<ImuData> imu_buffer_` 링버퍼 추가 (최대 500개, ~1.25초 @400Hz). ZED IMU ~400Hz, VO ~30fps → 프레임당 ~13개 IMU 샘플 버퍼링.
- **imuCallback/ZED SDK**: 두 경로 모두 `imu_buffer_.push_back()` 추가.
- **소비 코드 drain**: fusion 직전 `imu_buffer_` → `std::vector<ImuData>` 복사 후 clear. 각 샘플에 ZED→ROS 좌표 변환 적용.
- **fuse() 오버로드**: `ImuFusionBase`에 `fuse(PoseInput, ImuData, dt, vector<ImuData>)` 추가. 기본 구현은 단일 샘플 fallback (ComplementaryFilter/EKF 호환).
- **검증 로그**: `IMU buffer: N samples drained` (5초 throttle). N>0이면 버퍼링 정상.
- **수정 파일**: `vo_node.hpp`, `vo_node.cpp`, `imu_fusion.hpp`

### 2026-03-20 — Factor Graph 발산 버그 수정

- **Prior 노이즈 비율 수정**: prior_noise sigma `1e-6` → rotation `0.01 rad`, translation `0.05 m`. 기존 between_noise(0.05/0.1)와 50,000:1 비율로 pose 0이 과도하게 고정되어, 20개 between factor의 미세 오차가 마지막 pose에 누적 → 일정한 `rot_diff=2.290 rad`(131도) 발산. 비율을 5:1~2:1로 정상화.
- **invertDelta 버그 수정**: `odom_delta.valid=false`(zero motion) 시 `computeDelta(prev,curr)`가 이미 T_prev_from_curr(GTSAM Between 측정값)인데 `invertDelta`로 불필요하게 반전하고 있었음. invertDelta 제거.
- **현재 상태**: factor_graph에 IMU factor 미구현 (`(void)imu`). VO between factor만 사용하므로 VO 대비 이점 없음. IMU preintegration factor 추가 필요.

### 2026-03-11 — VO 좌표변환 완료

- **포즈 누적 (ZED wrapper 정렬)**: solvePnP 출력 T_prev_from_curr를 역변환 없이 직접 사용. `T_global_ = T_global_ * T_cp`. ZED `getPosition(CAMERA)` delta와 동일 규약.
- **위치 추출**: T_0_from_curr에서 translation = curr 원점의 frame0 좌표. t_body = R_opt_to_body * t_curr_in_0.
- **회전 similarity 변환**: R_body = R_opt_to_body * R_0_from_curr.t() * R_opt_to_body.t(). optical 프레임 회전을 body로 변환, 정지 시 I → camera_link가 odom과 축 정렬.
- **TF 규약 수정**: odom→camera_link는 child→parent 변환. R_odom_from_camera_link = R_body.t() 발행. 회전 역방향 문제 해결.
- **yaw 추출**: tf2 getRPY와 동일하게 atan2(R(1,0), R(0,0)) 사용 (부호 반전 제거).

### 2026-03-24

- **IMU-VO Adaptive Noise (보조 수단)**: `computeImuVoConsistency()` — IMU delta vs VO delta 비교, 불일치 시 noise_scale 최대 10x. 기본 between_noise 상향 (0.05/0.1 → 0.08/0.15). 단, factor-level 조절만으로는 불충분 — feature-level outlier rejection이 표준 접근 (ORB-SLAM3/VINS 분석 결과).
- **로드맵 재구성**: 동적 물체 대응을 ORB-SLAM3/VINS-Mono 표준 기준으로 5-Phase 재설계. Layer 1(RANSAC inlier ratio) → Layer 2(재투영 에러 필터) → Layer 3(Huber kernel) → Layer 4(multi-view) → Layer 5(semantic mask).

### 2026-03-10

- **Factor Graph 버그 수정**: GTSAM Between(i,j) measured=T_i_from_j. odom_delta를 T_prev_from_curr로 수정 (기존 T_curr_from_prev 잘못 전달로 발산).
- **Factor Graph Phase 5 완료**: factor_graph 모드 통합 완료. PoseOutput/TF 발행, RViz odom→camera_link 궤적 확인.
- **Factor Graph Phase 4 완료**: factor_graph_window_size, setWindowSize, slide 시 T_base 누적·그래프 재구성, getNextIndex로 인덱스 동기화.
- **Factor Graph Phase 3 완료**: PoseInput.odom_delta(RelPose), processImages에서 T_curr_from_prev(optical→body) 계산, ImuFusionFactorGraph에서 odom_delta.valid 시 직접 사용.
- **Factor Graph Phase 2 완료**: factor_graph.hpp/cpp, ImuFusionFactorGraph, addPose/addOdometryFactor/optimize, runVerification, GTSAM linear/NoiseModel.h.
- **Factor Graph Phase 1 완료**: GTSAM(ros-humble-gtsam) 설치, CMakeLists/package.xml 의존성 추가, colcon build 성공.
- **IMU-VO fusion**: complementary filter (roll/pitch from IMU, yaw/pos from VO). EKF 15-state (p,v,rpy,gyro_bias,accel_bias): predict from IMU, update from VO pose. fusion_mode: none|complementary|ekf|factor_graph. IMU ZED→ROS REP 103 변환 (az,-ax,-ay), (gz,-gx,-gy).
- **구독/발행 조건부**: zed_sdk 모드에서 rgb/depth/camera_info/imu 구독 미생성. ros2 모드에서 imu_pub 미생성.
- **enable_pose_estimation**: 기본값 true (vo_params.yaml).
- **IMU 발행**: zed_sdk 모드에서 getSensorsData(TIME_REFERENCE::IMAGE) → sensor_msgs/Imu. angular_velocity deg/s→rad/s, linear_acceleration m/s². topics.imu, imu.enable 파라미터. ZED2/ZED Mini만 is_available.
- **3D 점 생성**: curr_points + curr_depth → backprojectAndFilter. ZED depth mm 단위 (50~20000). prev_depth NaN 대비 use_curr_points. 0 valid 시 원본 매칭 유지.
- **Performance Metrics**: num_3d_points 추가 (1초마다 출력).
- **버그 수정 #1, #2**: Source 로그 → get_parameter 직접 읽기. zed_sdk camera_params_ → connect 후 getCameraParameters()+getResolution() 호출.
- **RGB-D 동기화 (ros2/zed_sdk)**: ros2 모드에서 rgbCallback이 current_depth_를 processImages에 전달. depth_mutex_로 동시 접근 보호. zed_sdk 모드에서는 rgb/depth 콜백 early return, 큐 경로만 사용.
- **DEV.md 생성**: 개발서 초안 작성
- **camera_params 초기화**: `logSystemInfo` 시 camera_width/height/fx/fy/cx/cy 미초기화로 쓰레기 값 출력 → 0으로 초기화, N/A 메시지 출력으로 수정
- **QoS 변경**: camera_info, depth 구독을 `SensorDataQoS()`로 변경 (ZED 호환)
- **camera_info 로그**: 수신 시 `Camera parameters received: WxH, fx=... fy=...` 출력 추가
- **publishResults**: camera_pose, vo_state, TF (odom→camera_link) 발행. frames.frame_id, frames.child_frame_id, tf.publish 파라미터.
- **좌표계**: ZED IMAGE→ROS REP 103. R_zed_to_ros, t_curr_in_f0=-R^T*t, tf2 quaternion.

---

## 11. 다음 작업 체크리스트

### 완료
- [x] Factor Graph Phase 5: factor_graph 통합, PoseOutput/TF 발행, RViz 궤적 확인
- [x] Factor Graph Phase 4: setWindowSize, T_base 누적, slide 시 그래프 재구성, getNextIndex
- [x] Factor Graph Phase 3: RelPose/odom_delta, T_curr_from_prev 직접 전달, PnP→Between factor
- [x] Factor Graph Phase 2: FactorGraphBackend, ImuFusionFactorGraph, runVerification
- [x] Factor Graph Phase 1: GTSAM 설치, CMakeLists/package.xml, 빌드 성공
- [x] rgbCallback에서 current_depth_를 processImages에 전달 (ros2/zed_sdk 모드 분리)
- [x] 3D 점 생성: curr_points + curr_depth → backprojectAndFilter (ZED mm: 50~20000)
- [x] PnP: solvePnPRansac 호출 + R,t 로그
- [x] 포즈 누적: T_global (ZED wrapper 정렬, T_prev_from_curr 직접 사용)
- [x] publishResults: camera_pose, vo_state, TF (frames.*, tf.publish 파라미터)
- [x] 좌표계: optical→body, TF R_odom_from_camera_link, camera_link 축 정렬
- [x] zero_motion 회전 체크
- [x] enable_pose_estimation 연동
- [x] IMU 발행 (zed_sdk)
- [x] Complementary filter
- [x] EKF 15-state

### 장기 Odometry 로드맵 (RESEARCH_SURVEY.md 순서)

1. **Factor Graph + Sliding Window** (단계별 구현·검증)

   > **워크플로우**: 각 Phase 완료 후 `colcon build` + 실행 + 검증 통과 시에만 다음 Phase 진행.

   **Phase 1: 의존성** ✅ 완료
   - [x] 1.1 GTSAM 설치 확인 (`apt install ros-humble-gtsam`)
   - [x] 1.2 CMakeLists에 `find_package(GTSAM)` 추가, package.xml 의존성 추가
   - [x] 1.3 빌드 성공 확인 (`colcon build`)

   **Phase 2: 최소 pose graph (독립 테스트)** ✅ 완료
   - [x] 2.1 `factor_graph.hpp` 생성: `FactorGraphBackend` 클래스 스켈레톤
   - [x] 2.2 `addPose(i, x,y,z, roll,pitch,yaw)`, `addOdometryFactor(i, j, delta_pose)` 인터페이스
   - [x] 2.3 `optimize()` → 최신 pose 반환
   - [x] 2.4 `ImuFusionFactorGraph` 연동, fusion_mode=factor_graph 시 FactorGraphBackend 사용
   - [x] 2.5 **검증**: `runVerification()` 3 pose + 2 edge → optimize, vo_node 시작 시 로그

   **Phase 3: VO → Factor 변환** ✅ 완료
   - [x] 3.1 `processImages`에서 PnP 성공 시 `T_curr_from_prev`(inv(T_prev_from_curr))를 PoseInput.odom_delta로 전달
   - [x] 3.2 Between factor: odom_delta.valid 시 PnP 측정값 직접 사용, else computeDelta
   - [x] 3.3 매 프레임: addPose + addOdometryFactor (이미 Phase 2)
   - [x] 3.4 **검증**: factor_graph 모드 실행, PnP OK, pose 추적 정상

   **Phase 4: Sliding window** ✅ 완료
   - [x] 4.1 윈도우 크기 N: `imu.factor_graph_window_size` (기본 20, 0=무제한)
   - [x] 4.2 N 초과 시 가장 오래된 pose 제거, T_base 누적, 그래프 재구성 (prior 재설정)
   - [x] 4.3 **검증**: factor_graph 모드 실행, sliding window 동작 확인

   **Phase 5: 통합** ✅ 완료
   - [x] 5.1 `fusion_mode: "factor_graph"` 시 ImuFusionFactorGraph → FactorGraphBackend.optimize() 출력을 pose로 사용
   - [x] 5.2 EKF/complementary와 동일한 PoseOutput 형식, publishResults(camera_pose, vo_state, TF)
   - [x] 5.3 **검증**: factor_graph 모드 실행, pose/TF 발행 정상 (RViz에서 odom→camera_link 확인 가능)

2. **루프 클로저** → §5로 이동 (동적 물체 대응 완료 후 진행)

3. **IMU Preintegration** (5-phase 구현)

   **Phase 1: IMU 버퍼링 인프라** ✅ 완료
   - [x] 1.1 `vo_node.hpp`: `std::deque<ImuData> imu_buffer_` + `kMaxImuBuffer=500`
   - [x] 1.2 `imuCallback` + ZED SDK 경로에서 `imu_buffer_.push_back()`
   - [x] 1.3 소비 코드: 버퍼 drain → `vector<ImuData>`, 각 샘플 좌표 변환
   - [x] 1.4 `ImuFusionBase::fuse()` 오버로드 (vector 인터페이스, 기본 fallback)
   - [x] 1.5 검증: `IMU buffer: N samples drained` 로그 (N > 0)

   **Phase 1.5: 고빈도 IMU 폴링** ✅ 완료
   - [x] 1.5.1 `ZEDInterface::getSensorsDataCurrent()` 추가 (`TIME_REFERENCE::CURRENT`)
   - [x] 1.5.2 `imuPollLoop()` 전용 스레드 200Hz (`vo_node.cpp`) — ROS2 타이머→스레드 변경 (executor 경합 해소)
   - [x] 1.5.3 `getImages()`에서 IMU 분리 (이미지/IMU 독립 획득)
   - [x] 1.5.4 검증: `IMU buffer: N samples drained` N ≈ 5~7 (실측 평균 6)

   **Phase 2: GTSAM IMU preintegration 설정** ✅ 완료
   - [x] 2.1 `PreintegrationCombinedParams` 파라미터 (acc/gyro noise, bias RW, gravity)
   - [x] 2.2 `preintegrateImu()`: `vector<ImuData>` → `PreintegratedCombinedMeasurements`
   - [x] 2.3 검증: `IMU preint: dt=... dp=(...) dv=(...) rpy=(...)deg` 로그 확인

   **Phase 3: Factor graph IMU factor 통합** 🔧 검증 중
   - [x] 3.1 velocity/bias 노드 추가 (`v(i)` Vector3, `b(i)` ConstantBias, prior)
   - [x] 3.2 `CombinedImuFactor` 추가 (BetweenFactor + IMU factor 동시)
   - [x] 3.3 `ImuFusionFactorGraph::fuse()` — preintegrate → imu_preintegrated flag → addImuFactor
   - [x] 3.4 sliding window에서 v/b 노드 shift + prior 재설정
   - [ ] 3.5 검증: factor_graph OK + `IMU factor added` 로그

   **Phase 4: 검증 & 튜닝**
   - [ ] 4.1 정지 상태 드리프트 비교 (VO-only vs VIO)
   - [ ] 4.2 동적 장면 드리프트 비교
   - [ ] 4.3 노이즈 파라미터 튜닝 (acc/gyro sigma)

   **Phase 5: IMU-VO Adaptive Noise** ✅ 구현 (보조 수단)
   - [x] 5.1 `computeImuVoConsistency()`: IMU delta vs VO delta 비교 → noise_scale 반환
   - [x] 5.2 `addOdometryFactor()`: noise_scale 파라미터, inflate 적용
   - [x] 5.3 기본 between_noise 상향 (0.08/0.15)
   - ⚠️ **한계**: factor-level noise 조절은 보조 수단. 단독으로는 불충분 (아래 §4 참조)

4. **동적 물체 Robust 대응**

   > **배경 및 교훈**:
   > - Factor-level noise 조절(IMU-VO consistency)은 정지 시에는 효과적이나,
   >   이동 중에는 프레임당 오염이 mm 단위로 작아 노이즈와 구분 불가 → 한계 도달
   > - RANSAC inlier ratio 체크는 물체가 FOV 대부분을 차지하면 RANSAC 자체가
   >   물체 운동에 fit → inlier ratio ≈100% → 작동 안 함
   > - **핵심 해결책**: IMU-guided feature filtering (OpenVINS chi² gating과 동일 접근)
   >   PnP 전에 IMU-predicted pose로 각 feature를 검증, 동적 물체 feature를 제거
   >
   > **방어 계층 (최종 설계)**:
   > ```
   > Layer 0: IMU-guided feature filtering  ← 핵심 (PnP 전 전처리) ★ 다음 구현
   > Layer 1: RANSAC PnP + inlier ratio     ← 기본 (구현 완료)
   > Layer 2: Huber robust kernel            ← factor graph (구현 완료)
   > Layer 3: IMU-VO consistency + ZUPT      ← 보조 (구현 완료, 튜닝 완료)
   > Layer 4: Semantic masking               ← 고급, 후순위
   > ```

   **Phase 1: RANSAC inlier ratio 기반 VO 신뢰도** ✅ 구현
   - [x] 1.1 `inlier_ratio = inliers / total_matches` 계산 (`frame_processor`)
   - [x] 1.2 `PoseInput.vo_confidence` 필드 → factor graph noise 조절
   - [x] 1.3 `< 0.5` noise_scale 증가, `< 0.3` VO 거의 폐기
   - ⚠️ **한계**: 물체가 FOV 대부분 차지 시 RANSAC이 물체에 fit → ratio ≈ 100%

   **Phase 2: Huber robust kernel on BetweenFactor** ✅ 구현
   - [x] 2.1 `noiseModel::Robust::Create(Huber(0.5), noise)` 적용

   **Phase 3: IMU-VO consistency 벡터 비교 + ZUPT 연동** ✅ 구현 (튜닝 완료)
   - [x] 3.1 `computeImuVoConsistency()`: 벡터 차이 비교 (방향+크기)
   - [x] 3.2 데드존 0.01m/0.03rad, 스케일 8x, 최대 20x
   - [x] 3.3 ZUPT+VO conflict: 정지 시 VO 5mm 이상 → noise_scale=20 (VO 폐기)
   - ⚠️ **한계**: 이동 중 프레임당 오염 mm 단위 → 노이즈와 구분 불가

   **Phase 4: IMU-guided feature filtering** (OpenVINS chi² gating) ✅ 완료
   > PnP **전에** IMU-predicted pose로 각 feature를 검증.
   > 동적 물체의 feature는 IMU 예측과 불일치 → 제거 → 깨끗한 feature로 PnP.
   > OpenVINS의 Mahalanobis gating과 수학적으로 동일한 접근.

   - [x] 4.1 IMU preintegration delta (R,t)를 frame_processor에 전달하는 인터페이스
     - `ImuPredictedPose` struct (frame_processor.hpp), `processFrame()` 파라미터 추가
   - [x] 4.2 IMU-predicted pose로 prev_3D점 재투영 (`cv::projectPoints`)
     - vo_node.cpp: IMU gyro/acc 적분 → body→optical 변환 → R,t 계산
   - [x] 4.3 재투영 에러 > threshold (5px)인 feature 제거
     - frame_processor.cpp: projected vs prev_points 비교, threshold 초과 제거
   - [x] 4.4 남은 feature로 PnP 실행 (오염 제거된 상태)
   - [x] 4.5 최소 feature 수 보장 (< 10개면 fallback: 기존 전체 PnP)
   - [x] 4.6 검증: `IMU-guided: removed N/M features` 로그 + 이동 중 드리프트 감소
     - 테스트 결과: 동적 물체 feature 49~94% 제거, 이동 중 드리프트 개선 확인

   **Phase 5: Multi-view consistency** (동적 물체 핵심 방어) ✅ 완료
   > 원리: 동적 물체의 feature는 RANSAC을 연속 N프레임 통과하기 어려움 → track이 짧음.
   > N프레임 이상 연속 추적된 feature만 PnP에 사용하면 동적 물체 feature 자연 제거.
   > ORB-SLAM3의 map point culling과 동일 접근 (관측 횟수 < threshold → 삭제).

   - [x] 5.1 `prev_track_ages_` 멤버: keypoint별 연속 추적 프레임 수 저장
   - [x] 5.2 매칭 후 age 계산: `curr_age[trainIdx] = prev_age[queryIdx] + 1`
   - [x] 5.3 age >= min_track_length(3) feature만 PnP에 사용
   - [x] 5.4 mature feature < 50개면 fallback (filtering 비활성)
   - [x] 5.5 로그: `multi-view: N mature / M total, removed K short-track`
   - [x] 5.6 검증: 이동 중 동적 물체 앞에서 드리프트 감소 (max drift 2.0m→0.8m, yaw ±170°→±78°)

   **Phase 5.5: GTSAM preintegration 기반 IMU prediction** (Phase 4 정확도 향상) ✅ 완료
   > 원리: raw gyro integration은 bias 미보정 → 33ms 이후에도 drift. GTSAM PIM은
   > factor graph에서 추정한 bias로 보정 + proper SO(3) integration → 더 정확한 rotation prediction.
   > 또한 velocity 상태를 활용한 translation prediction으로 constant velocity model 대체 가능.

   - [x] 5.5.1 `FactorGraphBackend::predictFromImu()`: temp PIM + prev_bias로 bias-corrected integration
   - [x] 5.5.2 `ImuFusionBase::predictFromImu()` virtual method (factor_graph 모드에서만 유효)
   - [x] 5.5.3 `ImuFusionFactorGraph::predictFromImu()` override → backend 호출
   - [x] 5.5.4 `vo_node.cpp`: GTSAM prediction 사용, raw gyro는 fallback으로 유지
   - [x] 5.5.5 Translation: PIM predict(velocity) 사용, ~0이면 constant velocity fallback
   - [x] 5.5.6 로그: `IMU predict: dt= rot= t= ang_rate= bias_g=`

   **Phase 6: Semantic masking** (고급, 후순위)
   - [ ] 6.1 경량 segmentation (MobileNet/YOLO)으로 사람/차량 마스크
   - [ ] 6.2 마스크 영역 feature 추출 제외

   > **비교표: 대표 시스템 vs 우리 시스템**
   > | 방어 계층 | OpenVINS | ORB-SLAM3 | RTAB-Map | 우리 시스템 |
   > |-----------|----------|-----------|----------|------------|
   > | RANSAC PnP | ✅ | ✅ | ✅ | ✅ |
   > | IMU-guided filtering | ✅ chi² | ❌ | ❌ | ✅ Phase 4 (GTSAM bias-corrected) |
   > | Multi-view consistency | ❌ | ✅ (map point culling) | ✅ | ✅ Phase 5 |
   > | Robust kernel | ❌ | ✅ (Huber) | ❌ | ✅ |
   > | IMU-VO noise 조절 | ❌ | ❌ | ❌ | ✅ |
   > | GTSAM preintegration prediction | ✅ | ❌ | ❌ | ✅ Phase 5.5 |
   > | Multi-round BA | ❌ | ✅ | ✅ | ❌ |
   > | Semantic mask | ❌ | ❌ | ❌ | ❌ |

5. **루프 클로저** (Phase 4 이후)
   - [ ] 키프레임 선택 (baseline: 0.3~0.5m 또는 15°)
   - [ ] Place recognition: BoW 또는 학습 기반
   - [ ] 루프 감지 시 constraint 추가 → Pose graph optimization
   - [ ] 장거리 누적 드리프트 보정

6. **실험·평가** (병행)
   - [ ] 데이터셋 수집 또는 TUM/EuRoC
   - [ ] ATE, RPE, 처리 속도 측정

### 선택 (후순위)
- [ ] 루프 클로저
- [ ] 3DGS/NeRF SLAM 연구 추적
- [ ] 스케일 보정, 번들 조정

---

## 12. 논문용 메모 (실험/결과 추후 기록)

- **실험 환경**: (하드웨어, OS, ROS 버전 등)
- **데이터셋**: (사용한 시퀀스, 경로)
- **평가 지표**: (ATE, RPE, 처리 속도 등)
- **결과**: (표, 그래프용 수치)

---

## 13. 수정 근거 (Rationale)

> 문서/코드 수정 시 왜 그렇게 했는지 근거를 기록. 코드 위치·논리·참고를 명시.

### §2 파이프라인 6번: curr_points + curr_depth

| 수정 | 근거 |
|------|------|
| `prev_points + prev_depth` → `curr_points + curr_depth` | **코드**: `frame_processor.cpp` L39-43 `backprojectAndFilter(..., use_curr=true)`. **논리**: prev_depth는 ZED/ros2에서 NaN이거나 미동기화될 수 있음 (§8 버그 #3). curr_depth는 processImages 시점에 전달되므로 신뢰 가능. **참고**: `sample_points = use_curr_points ? curr_points : prev_points` (L94). |

### §3.2 FrameProcessor 파이프라인: backprojectAndFilter

| 수정 | 근거 |
|------|------|
| `match` → `match → backprojectAndFilter` | **코드**: `frame_processor.cpp` L34-44. match 후 `backprojectAndFilter(result.matches, depth, camera_params, true)` 호출. **논리**: 2D 매칭만으로는 PnP 불가. 3D 점 생성이 필수. **참고**: `frame_processor.hpp` L41-44. |

### §3.3 데이터 흐름: backprojectAndFilter(curr_points+depth)

| 수정 | 근거 |
|------|------|
| `backprojectAndFilter(curr_points+depth)` 명시 | **코드**: `frame_processor.cpp` L94 `sample_points = use_curr_points ? curr_points : prev_points`, 호출 시 `use_curr= true`. **논리**: §2 6번과 동일. curr 2D + curr depth → 3D. |

### §4 FeatureMatches.prev_points_3d

| 수정 | 근거 |
|------|------|
| `prev_points_3d (curr 3D, PnP용)` | **코드**: `types.hpp` L43, `frame_processor.cpp` L154 `matches.prev_points_3d = std::move(valid_3d)`. valid_3d는 curr_points + curr_depth backproject 결과. **논리**: 변수명은 prev_points와 짝을 이루는 3D이지만, 실제 값은 curr 카메라 좌표계 3D. PnP objectPoints로 사용. |

### §4 ProcessingResult.visualization_time, is_keyframe

| 수정 | 근거 |
|------|------|
| `visualization_time`, `is_keyframe` 추가 | **코드**: `frame_processor.hpp` L21-22 선언, `vo_node.cpp` L521 `metrics.visualization_time = ...`, `logger.hpp` L46, `logger.cpp` L46 출력. **논리**: 메트릭 로깅·디버깅용. is_keyframe은 향후 키프레임 기반 최적화용 예약. |

### §11 PnP 체크리스트: prev_points_3d, prev_points

| 수정 | 근거 |
|------|------|
| `objectPoints=prev_points_3d`, `imagePoints=prev_points` | **코드**: `frame_processor.cpp` L39 주석 "objectPoints=curr 3D, imagePoints=prev 2D". **논리**: OpenCV solvePnP: objectPoints=3D (월드/객체 좌표), imagePoints=2D 투영. VO에서 curr 3D + prev 2D → R,t (curr→prev). prev_points_3d는 curr 카메라 좌표계 3D이므로 objectPoints로 사용. |

### 디버깅 체크리스트: zed_sdk/ros2, 3D Points

| 수정 | 근거 |
|------|------|
| zed_sdk vs ros2 구분 | **코드**: `vo_node.cpp` input.source 분기, `zed_interface.cpp` vs 콜백 경로. **논리**: zed_sdk는 USB 직접, ros2는 wrapper 필수. 연결/실행 경로가 다름. |
| `3D Points: N > 0` | **코드**: `logger.cpp` Performance Metrics, `num_3d_points` 출력. **논리**: backprojectAndFilter 후 valid 3D가 0이면 PnP 불가. depth 품질·범위(50~20000mm) 확인용. |

### §8 버그 #3: prev_depth 우회

| 수정 | 근거 |
|------|------|
| curr_depth + curr_points 사용 | **코드**: `frame_processor.cpp` L41-43, L94. **논리**: prev_depth는 ros2에서 별도 콜백으로 오며 NaN/미동기화 가능. curr_depth는 processImages 시 rgb와 함께 전달되므로 동기화·유효성 보장. |

### 포즈 누적 T_global (구: 역변환 방식 → 2026-03-11 ZED 방식으로 변경)

| 수정 | 근거 |
|------|------|
| T_global = T_global * T_cp (T_prev_from_curr) | **코드**: `vo_node.cpp` L617-620. **논리**: solvePnP = T_prev_from_curr. ZED wrapper와 동일하게 직접 사용. **단위**: mm → /1000으로 m. |

### 좌표계 (optical→body, camera_link + static)

| 항목 | 근거 |
|------|------|
| odom→camera_link | body frame pose. R_opt_to_body로 변환. TF는 R_odom_from_camera_link 발행. |
| camera_link→camera_optical_frame | static TF (0,0,0, R_opt_to_body). REP 103 표준. |
| pose | t_curr_in_0 직접 사용, t_body=R_opt_to_body*t_opt. R_body similarity 변환 후 R_tf=R_body.t() 발행. |

### 포즈 누적 T_prev_from_curr (2026-03-11)

| 수정 | 근거 |
|------|------|
| T_prev_from_curr 직접 사용 (역변환 제거) | **코드**: `vo_node.cpp` L617-620. **참고**: ZED wrapper `mOdom2BaseTransf = mOdom2BaseTransf * deltaOdomTf`. solvePnP(curr_3d, prev_2d) = T_prev_from_curr. |
| R_body similarity 변환 | **코드**: `vo_node.cpp` L639-640. **논리**: R_0_from_curr는 optical. body 프레임으로 변환 시 정지 시 I 필요. R_body = R_opt_to_body * R_0_from_curr.t() * R_opt_to_body.t(). |
| R_tf = R_body.t() | **코드**: `vo_node.cpp` L641-643. **논리**: TF는 p_odom = R_odom_from_camera_link * p_camera_link. R_body = R_camera_link_from_odom이므로 역행렬 발행. |
