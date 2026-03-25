#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>


namespace vo {

/** GTSAM preintegration 기반 IMU prediction 결과 (body frame) */
struct ImuPrediction {
    double R[9]{1,0,0, 0,1,0, 0,0,1};  // 3x3 row-major rotation delta (body frame)
    double tx{0}, ty{0}, tz{0};          // translation delta (body frame, m) — PIM predict w/ velocity
    double angular_rate{0};              // rad/s — adaptive threshold용
    double total_dt{0};                  // integration time (sec)
    bool valid{false};
};

/** EKF robust params (Chi-squared gating + Huber) - VINS/ORB-SLAM 계열 표준 */
struct EKFParams {
    double chi2_threshold{16.8};   // χ²(6, 0.99) ≈ 16.8
    double huber_pos_m{0.1};       // position innovation clip (m)
    double huber_rot_rad{0.1};     // rotation innovation clip (rad)
};

/** IMU raw data (ROS convention: rad/s, m/s²). Must be in same frame as VO pose (ROS body). */
struct ImuData {
    double ang_vel_x{0.0}, ang_vel_y{0.0}, ang_vel_z{0.0};  // rad/s
    double lin_acc_x{0.0}, lin_acc_y{0.0}, lin_acc_z{0.0};  // m/s²
    double timestamp{0.0};  // sec
    bool valid{false};
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
