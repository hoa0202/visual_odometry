#pragma once

#include "visual_odometry/imu_fusion.hpp"
#include <cstddef>

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

    /** odometry factor: Between(i,j), measured = T_i_from_j (delta_pose) */
    void addOdometryFactor(size_t i, size_t j, const DeltaPose& delta_pose);

    /** 최적화 후 최신 pose 반환. pose가 없으면 identity. */
    PoseOutput optimize();

    /** 그래프 초기화 */
    void reset();

    /** Phase 2.5 검증: 3 pose + 2 edge → optimize → pose 로그. 성공 시 true. */
    static bool runVerification();

    /** prev→curr 상대 pose (T_curr_from_prev). addOdometryFactor에는 T_prev_from_curr 필요. */
    static DeltaPose computeDelta(const PoseOutput& prev, const PoseOutput& curr);
    /** DeltaPose 역변환 (T_curr_from_prev → T_prev_from_curr) */
    static DeltaPose invertDelta(const DeltaPose& d);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vo
