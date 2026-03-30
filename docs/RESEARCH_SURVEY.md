# Visual-Inertial Odometry / SLAM 연구 서베이

> 2024–2025 전망 있는 연구 방향 정리. 프로젝트(장기 odometry, 루프 클로저, factor graph)와 연관된 논문·트렌드 중심.

---

## 1. Factor Graph / Sliding Window 최적화

| 논문/시스템 | 핵심 | 전망 |
|-------------|------|------|
| **Sliding-Window FGO** | 고정 윈도우 내 pose+측정 동시 최적화, marginalization으로 실시간 유지 | 실시간 VIO의 표준 구조. GTSAM/g2o/Ceres로 구현 가능 |
| **RFG-TVIU** (Frontiers 2024) | Vision/IMU/UWB tightly-coupled factor graph, adaptive robust factor | VIO 단독 대비 RMSE 62–82% 개선. 다중 센서 확장 모델 |
| **Real-time GNSS/IMU FGO** (arXiv 2603) | loosely coupled GNSS+IMU factor graph, 실시간 | 실외/GNSS 활용 시 참고 |
| **MATLAB Factor Graph VIO** | monocular VIO factor graph, partial graph optimization | GTSAM 기반 구현 참고용 |

**정리**: Factor graph + sliding window는 EKF를 대체하는 실시간 백엔드로 적합. 루프 클로저 edge 추가가 자연스럽다.

---

## 2. 루프 클로저 / Place Recognition

| 논문/시스템 | 핵심 | 전망 |
|-------------|------|------|
| **PRAM** (arXiv 2404) | Transformer 기반 place recognition, sparse keypoint에서 landmark 인식 | 기존 대비 2.4× 빠름, 저장량 90% 이상 감소 |
| **Loopy-SLAM** (CVPR 2024) | Dense neural SLAM + global place recognition, frame-to-model tracking | RGB-D, neural 기반 루프 클로저 |
| **AnyLoc 통합** (arXiv 2601) | BoVW 대신 AnyLoc deep feature, adaptive threshold | viewpoint/조명 변화에 강함 |
| **Robust Loop Closure** (MDPI 2024) | challenging 환경용 lightweight loop closure | 저사양/엣지 디바이스 고려 |

**정리**: 학습 기반 place recognition이 BoW를 대체하는 추세. PRAM, AnyLoc 등이 경량·효율 측면에서 유력.

---

## 3. Visual-Inertial SLAM 시스템

| 논문/시스템 | 핵심 | 전망 |
|-------------|------|------|
| **OKVIS2-X** (2025) | Keyframe 기반 VI-SLAM, dense depth/LiDAR/GNSS 선택 가능 | EuRoC·Hilti22에서 SOTA, 모듈형 구조 |
| **ORB-SLAM3** | Multi-map, Atlas, VI tightly-coupled | 실시간 로봇 네비게이션용으로 검증됨 |
| **VINS-Fusion** | Tightly coupled VI, factor graph | ORB-SLAM3와 비슷한 수준, VI 특화 |
| **OpenVINS** | Modular VI, reduced-scale vehicle benchmark에서 최저 오차 | 연구·실험용으로 적합 |
| **DROID-SLAM** | Learning-based, dense | 전통 방식과 다른 접근 |
| **Kimera, SVO Pro** | Semantic/robust VI | 특수 용도 참고 |

**정리**: ORB-SLAM3, VINS-Fusion, OKVIS2-X가 실용·벤치마크 측면에서 우선 참고 대상.

---

## 4. 3D Gaussian Splatting + SLAM

| 논문/시스템 | 핵심 | 전망 |
|-------------|------|------|
| **VIGS-SLAM** (2024) | VI + 3D Gaussian Splatting, motion blur/저조도 대응 | 3DGS 기반 SLAM의 대표 사례 |
| **MASt3R-SLAM** (arXiv 2412) | 3D reconstruction prior, dense SLAM, 15 FPS | 2차 global optimization, dense geometry |
| **NeRF/3DGS SLAM 서베이** | hand-crafted → deep learning → radiance field | 장기적으로는 신규 표현 방식 |

**정리**: 3DGS/NeRF SLAM은 연구 단계. 실시간·장기 odometry에는 factor graph + 기존 feature가 우선.

---

## 5. 다중 센서 융합

| 트렌드 | 내용 |
|--------|------|
| **LiDAR-Visual-IMU** | LVI-SAM, FAST-LIVO 등, 실외/대규모 환경 |
| **GNSS-VI** | OKVIS2-X, 실외 global consistency |
| **UWB-VI** | RFG-TVIU, 실내 정밀 위치 보정 |

**정리**: ZED만 사용하면 VI에 집중. LiDAR/GNSS 등이 있으면 다중 센서 융합 논문 참고.

---

## 6. 벤치마크·평가

| 출처 | 내용 |
|------|------|
| **TUM RGB-D** | ORB-SLAM3 ATE 0.0091 m, VINS-Fusion 0.0115 m |
| **EuRoC** | OKVIS2-X SOTA |
| **Hilti22** | OKVIS2-X VI-only 상위 |
| **outdoor benchmark** (arXiv 2408) | 루프 클로저의 정확도·연산 비용 분석 |

---

## 7. 프로젝트 적용 우선순위

| 순서 | 항목 | 근거 |
|------|------|------|
| 1 | **Factor Graph + Sliding Window** | EKF 대체, 루프 클로저 확장 용이 |
| 2 | **루프 클로저 (BoW 또는 학습)** | 장기 드리프트 감소, PRAM/AnyLoc 등 경량 |
| 3 | **IMU Preintegration** | GTSAM 기반 VI에서 표준 |
| 4 | **3DGS/NeRF** | 연구 추적용, 구현은 후순위 |

---

## 8. 참고 자료 링크

- **OKVIS2-X**: Semantic Scholar
- **PRAM**: arXiv 2404.07785
- **Loopy-SLAM**: arXiv 2402.09944, CVPR 2024
- **VIGS-SLAM**: arXiv 2512.02293
- **Outdoor VI-SLAM benchmark**: arXiv 2408.01716
- **RFG-TVIU**: Frontiers in Neurorobotics 2024
- **SLAM Survey (Frontiers 2024)**: frontiersin.org/articles/10.3389/frobt.2024.1347985
