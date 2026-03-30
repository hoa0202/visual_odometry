#pragma once

#include "visual_odometry/imu_fusion.hpp"
#include <cstddef>
#include <vector>

namespace vo {

/** 6-DOF relative pose (x,y,z m; roll,pitch,yaw rad). GTSAM Between(i,j) expects T_i_from_j. */
struct DeltaPose {
    double x{0.0}, y{0.0}, z{0.0};
    double roll{0.0}, pitch{0.0}, yaw{0.0};
};

/** GTSAM 기반 pose graph 백엔드. VO odometry를 factor로 누적·최적화. */
class FactorGraphBackend {
public:
    FactorGraphBackend();
    ~FactorGraphBackend();

    /** Phase 4: sliding window 크기. 0=무제한. 기본 20. */
    void setWindowSize(size_t N);

    /** 다음 pose 인덱스 (slide 후 재사용). fusion에서 addPose/addOdometryFactor 호출 전 사용. */
    size_t getNextIndex() const;

    /** pose 노드 추가 (초기값). i=0은 prior로 고정. */
    void addPose(size_t i, double x, double y, double z,
                 double roll, double pitch, double yaw);

    /** odometry factor: Between(i,j), measured = T_i_from_j (delta_pose).
     *  noise_scale > 1.0 → VO를 덜 신뢰 (IMU-VO 불일치 시 inflate). */
    void addOdometryFactor(size_t i, size_t j, const DeltaPose& delta_pose,
                           double noise_scale = 1.0);

    /** IMU preintegration delta와 VO delta의 consistency score 계산.
     *  반환: noise_scale (1.0=일치, 최대 10.0=심한 불일치). */
    double computeImuVoConsistency(const DeltaPose& vo_delta) const;

    /** 최적화 후 최신 pose 반환. pose가 없으면 identity. */
    PoseOutput optimize();

    /** 그래프 초기화 */
    void reset();

    /** Phase 2.5 검증: 3 pose + 2 edge → optimize → pose 로그. 성공 시 true. */
    static bool runVerification();

    /** IMU preintegration 초기화 (Phase 2) */
    void initImuPreintegration(const ImuPreintegrationParams& imu_params);

    /** IMU 샘플 preintegrate. 프레임 간 IMU 버퍼를 전달. */
    void preintegrateImu(const std::vector<ImuData>& imu_samples);

    /** preintegration 결과 로그 (delta_p, delta_v, delta_R, dt) */
    void logPreintegration() const;

    /** preintegration 초기화 여부 */
    bool isImuReady() const;

    /** Phase 3: IMU factor를 그래프에 추가 (CombinedImuFactor).
     *  addPose() 후, addOdometryFactor()와 함께 호출.
     *  velocity/bias 초기값 자동 삽입. */
    void addImuFactor(size_t i, size_t j, double dt_sec);

    /** Phase 3: velocity/bias 초기값 설정 (i=0 prior용) */
    void addVelocityBiasPrior(size_t i);

    /** Phase 4: Zero-Velocity Update (ZUPT).
     *  IMU 기반 정지 감지 시 호출. zero-velocity prior + identity BetweenFactor 추가.
     *  VO 드리프트를 IMU로 억제. */
    void addZeroVelocityConstraint(size_t i);

    /** GTSAM preintegration 기반 IMU prediction (bias-corrected).
     *  별도 temp PIM으로 body-frame IMU 샘플 적분 → deltaR, predict(velocity) → deltaP.
     *  factor graph의 prev_bias 사용으로 raw gyro보다 정확. */
    ImuPrediction predictFromImu(const std::vector<ImuData>& body_frame_samples) const;

    /** IMU 샘플로 정지 상태 판별.
     *  gyro < threshold && accel ≈ gravity → true. */
    static bool detectZeroMotion(const std::vector<ImuData>& imu_samples,
                                  double gyro_threshold = 0.05,
                                  double accel_var_threshold = 0.15);

    /** Phase D: Camera calibration 설정 (reprojection factor용) */
    void setCameraCalibration(double fx, double fy, double cx, double cy);

    /** Phase D: Reprojection factors 추가 (sliding window BA).
     *  pose_idx: 현재 pose 인덱스.
     *  track_ids: KLT track IDs. pixels: 2D 관측 (u,v). points_3d_cam: camera frame 3D (mm→m 변환 내부 수행).
     *  world frame 3D landmark 생성/갱신 + GenericProjectionFactor 추가. */
    void addReprojectionFactors(size_t pose_idx,
                                const std::vector<int>& track_ids,
                                const std::vector<cv::Point2f>& pixels,
                                const std::vector<cv::Point3f>& points_3d_cam);

    /** ORB-SLAM3 style Motion-only BA.
     *  현재 pose 1개만 최적화. Map points는 고정. Between factor 없음.
     *  optical frame에서 직접 동작 (body frame 변환 없음).
     *  @param T_world_cam  4x4 T_world_from_camera (optical frame, mm)
     *  @param world_points  map에서 가져온 3D world points (optical frame, mm)
     *  @param pixels        대응하는 2D 관측 (pixel)
     *  @param out_inliers   출력: 최종 inlier 수
     *  @return 최적화된 4x4 T_world_from_camera. 실패 시 입력 그대로. */
    /** @param octaves  각 pixel의 pyramid octave level (information matrix 스케일링용).
     *                   비어있으면 전부 level 0 취급. */
    cv::Mat motionOnlyBA(const cv::Mat& T_world_cam,
                         const std::vector<cv::Point3f>& world_points,
                         const std::vector<cv::Point2f>& pixels,
                         int& out_inliers,
                         const std::vector<int>& octaves = {}) const;

    /** prev→curr 상대 pose (T_curr_from_prev). addOdometryFactor에는 T_prev_from_curr 필요. */
    static DeltaPose computeDelta(const PoseOutput& prev, const PoseOutput& curr);
    /** DeltaPose 역변환 (T_curr_from_prev → T_prev_from_curr) */
    static DeltaPose invertDelta(const DeltaPose& d);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vo
