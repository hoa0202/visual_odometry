#pragma once

#include "visual_odometry/imu_fusion.hpp"
#include <cstddef>
#include <vector>

namespace vo {

/** IMU preintegration 노이즈 파라미터 (ZED 2i BMI088 기본값) */
struct ImuPreintegrationParams {
    double accel_noise_sigma{0.05};       // m/s²/√Hz
    double gyro_noise_sigma{0.005};       // rad/s/√Hz
    double accel_bias_rw_sigma{0.001};    // m/s³/√Hz (bias random walk)
    double gyro_bias_rw_sigma{0.0001};    // rad/s²/√Hz
    double gravity{9.81};                 // m/s² (Z-up, ROS convention)
};

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

    /** IMU 샘플로 정지 상태 판별.
     *  gyro < threshold && accel ≈ gravity → true. */
    static bool detectZeroMotion(const std::vector<ImuData>& imu_samples,
                                  double gyro_threshold = 0.05,
                                  double accel_var_threshold = 0.15);

    /** prev→curr 상대 pose (T_curr_from_prev). addOdometryFactor에는 T_prev_from_curr 필요. */
    static DeltaPose computeDelta(const PoseOutput& prev, const PoseOutput& curr);
    /** DeltaPose 역변환 (T_curr_from_prev → T_prev_from_curr) */
    static DeltaPose invertDelta(const DeltaPose& d);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vo
