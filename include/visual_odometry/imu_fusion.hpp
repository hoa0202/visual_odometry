#pragma once

#include <cmath>
#include <memory>
#include <string>

namespace vo {

/** IMU raw data (ROS convention: rad/s, m/s²). Must be in same frame as VO pose (ROS body). */
struct ImuData {
    double ang_vel_x{0.0}, ang_vel_y{0.0}, ang_vel_z{0.0};  // rad/s
    double lin_acc_x{0.0}, lin_acc_y{0.0}, lin_acc_z{0.0};  // m/s²
    double timestamp{0.0};  // sec
    bool valid{false};
};

/** VO pose input (ROS frame: x,y,z m; roll,pitch,yaw rad) */
struct PoseInput {
    double x{0.0}, y{0.0}, z{0.0};
    double roll{0.0}, pitch{0.0}, yaw{0.0};
    bool valid{false};
};

/** Fused pose output */
struct PoseOutput {
    double x{0.0}, y{0.0}, z{0.0};
    double roll{0.0}, pitch{0.0}, yaw{0.0};
};

/** Base interface for IMU-VO fusion */
class ImuFusionBase {
public:
    virtual ~ImuFusionBase() = default;
    virtual PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) = 0;
    virtual void reset() = 0;
};

/** Complementary filter: roll/pitch from IMU (accel+gyro), yaw/pos from VO */
class ComplementaryFilter : public ImuFusionBase {
public:
    explicit ComplementaryFilter(double alpha = 0.98) : alpha_(alpha) {}

    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) override {
        PoseOutput out;
        out.x = vo_pose.x;
        out.y = vo_pose.y;
        out.z = vo_pose.z;
        out.yaw = vo_pose.yaw;  // VO only (vision)

        if (imu.valid) {
            // Accel-based roll/pitch (gravity direction)
            double ax = imu.lin_acc_x, ay = imu.lin_acc_y, az = imu.lin_acc_z;
            double norm = std::sqrt(ax*ax + ay*ay + az*az);
            if (norm > 0.1) {  // avoid div by zero when stationary
                double acc_roll = std::atan2(ay, std::sqrt(ax*ax + az*az));
                double acc_pitch = std::atan2(-ax, std::sqrt(ay*ay + az*az));

                // Complementary: alpha * (gyro_integrated) + (1-alpha) * accel
                roll_ += imu.ang_vel_x * dt_sec;
                pitch_ += imu.ang_vel_y * dt_sec;

                out.roll = alpha_ * roll_ + (1.0 - alpha_) * acc_roll;
                out.pitch = alpha_ * pitch_ + (1.0 - alpha_) * acc_pitch;

                roll_ = out.roll;
                pitch_ = out.pitch;
            } else {
                out.roll = roll_;
                out.pitch = pitch_;
            }
        } else {
            out.roll = vo_pose.roll;
            out.pitch = vo_pose.pitch;
            roll_ = vo_pose.roll;
            pitch_ = vo_pose.pitch;
        }
        return out;
    }

    void reset() override {
        roll_ = 0.0;
        pitch_ = 0.0;
    }

private:
    double alpha_;
    double roll_{0.0}, pitch_{0.0};
};

/** EKF 15-state: prediction from IMU, update from VO (stub: returns VO) */
class ImuFusionEKF : public ImuFusionBase {
public:
    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) override {
        (void)imu;
        (void)dt_sec;
        PoseOutput out;
        out.x = vo_pose.x;
        out.y = vo_pose.y;
        out.z = vo_pose.z;
        out.roll = vo_pose.roll;
        out.pitch = vo_pose.pitch;
        out.yaw = vo_pose.yaw;
        return out;  // TODO: EKF implementation
    }
    void reset() override {}
};

/** Factor graph + IMU preintegration (stub: returns VO) */
class ImuFusionFactorGraph : public ImuFusionBase {
public:
    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) override {
        (void)imu;
        (void)dt_sec;
        PoseOutput out;
        out.x = vo_pose.x;
        out.y = vo_pose.y;
        out.z = vo_pose.z;
        out.roll = vo_pose.roll;
        out.pitch = vo_pose.pitch;
        out.yaw = vo_pose.yaw;
        return out;  // TODO: Factor graph implementation
    }
    void reset() override {}
};

/** Factory */
inline std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha = 0.98) {
    if (mode == "complementary") {
        return std::make_unique<ComplementaryFilter>(alpha);
    }
    if (mode == "ekf") {
        return std::make_unique<ImuFusionEKF>();
    }
    if (mode == "factor_graph") {
        return std::make_unique<ImuFusionFactorGraph>();
    }
    return nullptr;
}

}  // namespace vo
