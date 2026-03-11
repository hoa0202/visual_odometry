#pragma once

#include "visual_odometry/imu_fusion.hpp"
#include <opencv2/core.hpp>

namespace vo {

/** EKF 15-state: p(3), v(3), rpy(3), gyro_bias(3), accel_bias(3) */
class ImuFusionEKF : public ImuFusionBase {
public:
    explicit ImuFusionEKF(const EKFParams& params = EKFParams{});

    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) override;
    void reset() override;

private:
    bool update(const PoseInput& vo_pose);  // returns true if applied

    void predict(const ImuData& imu, double dt);
    void predictOrientationOnly(const ImuData& imu, double dt);  // PnP fail 시 position hold

    // Euler rate from body angular velocity [wx,wy,wz]
    cv::Mat eulerRateFromOmega(double roll, double pitch, double yaw,
                               double wx, double wy, double wz) const;
    cv::Mat eulerToR(double roll, double pitch, double yaw) const;

    static constexpr int N = 15;
    cv::Mat x_;   // state 15x1
    cv::Mat P_;   // covariance 15x15
    bool initialized_{false};
    double p_prev_[3]{0, 0, 0};  // for velocity-from-VO
    bool p_prev_valid_{false};
    EKFParams params_;

    static constexpr double g_ = 9.81;
};

}  // namespace vo
