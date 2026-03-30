#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"


namespace vo {

/** GTSAM preintegration 기반 IMU prediction 결과 (body frame) */
struct ImuPrediction {
    double R[9]{1,0,0, 0,1,0, 0,0,1};
    double tx{0}, ty{0}, tz{0};
    double angular_rate{0};
    double total_dt{0};
    bool valid{false};
};

struct EKFParams {
    double chi2_threshold{16.8};
    double huber_pos_m{0.1};
    double huber_rot_rad{0.1};
};

/** Relative pose T_curr_from_prev (for factor graph Between factor) */
struct RelPose {
    double x{0.0}, y{0.0}, z{0.0};
    double roll{0.0}, pitch{0.0}, yaw{0.0};
    bool valid{false};
};

/** VO pose input (ROS frame: x,y,z m; roll,pitch,yaw rad) */
struct PoseInput {
    double x{0.0}, y{0.0}, z{0.0};
    double roll{0.0}, pitch{0.0}, yaw{0.0};
    bool valid{false};
    /** PnP T_prev_from_curr in body frame (GTSAM Between measured). factor_graph만 사용. */
    RelPose odom_delta;
    /** RANSAC inlier ratio (0.0~1.0). 낮을수록 동적 물체 오염 가능성 높음. */
    double vo_confidence{1.0};
    /** Phase D: Reprojection factor용 데이터 (sliding window BA) */
    std::vector<int> track_ids;
    std::vector<cv::Point2f> pixels;
    std::vector<cv::Point3f> points_3d_cam;  // camera frame, mm
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
    /** IMU 버퍼 전달 오버로드 (factor_graph preintegration용). 기본: 단일 샘플 fallback. */
    virtual PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec,
                            const std::vector<ImuData>& imu_samples) {
        (void)imu_samples;
        return fuse(vo_pose, imu, dt_sec);
    }
    virtual void reset() = 0;
    /** GTSAM preintegration 기반 IMU prediction (bias-corrected rotation + velocity-based translation).
     *  body_frame_samples: ROS body frame IMU 데이터. factor_graph 모드에서만 유효. */
    virtual ImuPrediction predictFromImu(const std::vector<ImuData>& body_frame_samples) {
        (void)body_frame_samples;
        return {};  // default: no prediction (complementary/EKF)
    }
    /** Phase D: Camera calibration 설정 (reprojection factor용). factor_graph 모드에서만 유효. */
    virtual void setCameraCalibration(double fx, double fy, double cx, double cy) {
        (void)fx; (void)fy; (void)cx; (void)cy;
    }
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

/** Factor graph + IMU preintegration. Phase 2~4: pose graph, sliding window. */
class ImuFusionFactorGraph : public ImuFusionBase {
public:
    explicit ImuFusionFactorGraph(size_t window_size = 20);
    ~ImuFusionFactorGraph() override;
    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) override;
    PoseOutput fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec,
                    const std::vector<ImuData>& imu_samples) override;
    void reset() override;
    ImuPrediction predictFromImu(const std::vector<ImuData>& body_frame_samples) override;
    void setCameraCalibration(double fx, double fy, double cx, double cy) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha = 0.98);
std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha,
                                               const EKFParams& ekf_params);
std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha,
                                               const EKFParams& ekf_params,
                                               size_t factor_graph_window_size);

}  // namespace vo
