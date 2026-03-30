#include "visual_odometry/imu_fusion.hpp"
#include "visual_odometry/factor_graph.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <algorithm>

namespace vo {

static rclcpp::Logger logger() {
    return rclcpp::get_logger("factor_graph");
}
static rclcpp::Clock::SharedPtr throttle_clock() {
    static auto c = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    return c;
}

struct ImuFusionFactorGraph::Impl {
    FactorGraphBackend backend;
    PoseOutput prev_pose;
    bool has_prev{false};
    bool imu_preintegrated{false};  // 현재 프레임에서 preintegration 완료 여부
    bool is_zero_motion{false};     // Phase 4: IMU 기반 정지 감지
    bool camera_calib_set{false};   // Phase D: camera calibration 설정 여부
};

ImuFusionFactorGraph::ImuFusionFactorGraph(size_t window_size)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend.setWindowSize(window_size);
    // Phase 2: IMU preintegration 초기화 (기본 노이즈 파라미터)
    ImuPreintegrationParams imu_params;
    impl_->backend.initImuPreintegration(imu_params);
    RCLCPP_INFO(logger(), "IMU preintegration initialized (acc_sigma=%.3f gyro_sigma=%.4f)",
        imu_params.accel_noise_sigma, imu_params.gyro_noise_sigma);
}

ImuFusionFactorGraph::~ImuFusionFactorGraph() = default;

PoseOutput ImuFusionFactorGraph::fuse(const PoseInput& vo_pose,
                                       const ImuData& imu, double dt_sec) {
    (void)imu;
    (void)dt_sec;
    PoseOutput curr;
    curr.x = vo_pose.x;
    curr.y = vo_pose.y;
    curr.z = vo_pose.z;
    curr.roll = vo_pose.roll;
    curr.pitch = vo_pose.pitch;
    curr.yaw = vo_pose.yaw;

    if (!vo_pose.valid) {
        return impl_->has_prev ? impl_->prev_pose : curr;
    }

    size_t idx = impl_->backend.getNextIndex();
    impl_->backend.addPose(idx, curr.x, curr.y, curr.z,
                           curr.roll, curr.pitch, curr.yaw);

    // Phase 3: 첫 프레임에서 velocity/bias prior 추가
    if (idx == 0) {
        impl_->backend.addVelocityBiasPrior(0);
    }

    if (idx > 0) {
        DeltaPose delta;
        if (vo_pose.odom_delta.valid) {
            delta.x = vo_pose.odom_delta.x;
            delta.y = vo_pose.odom_delta.y;
            delta.z = vo_pose.odom_delta.z;
            delta.roll = vo_pose.odom_delta.roll;
            delta.pitch = vo_pose.odom_delta.pitch;
            delta.yaw = vo_pose.odom_delta.yaw;
        } else {
            delta = FactorGraphBackend::computeDelta(impl_->prev_pose, curr);
        }
        // Phase 5: IMU-VO consistency → adaptive noise
        double noise_scale = impl_->backend.computeImuVoConsistency(delta);

        // ZUPT + IMU-VO 연동: IMU가 정지인데 VO가 큰 motion → VO 거의 무시
        if (impl_->is_zero_motion) {
            double vo_motion = std::sqrt(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
            if (vo_motion > 0.005) {  // 5mm 이상 움직임 → 물체 오염
                noise_scale = std::max(noise_scale, 20.0);
                RCLCPP_WARN_THROTTLE(logger(), *throttle_clock(), 2000,
                    "ZUPT+VO conflict: IMU=stationary but VO=%.4fm → noise_scale=%.1f (VO discarded)",
                    vo_motion, noise_scale);
            }
        }

        // Layer 1: RANSAC inlier ratio 기반 VO 신뢰도 (ORB-SLAM3 핵심 방어선)
        double vo_conf = vo_pose.vo_confidence;
        if (vo_conf < 0.3) {
            noise_scale = std::max(noise_scale, 10.0);
            RCLCPP_WARN_THROTTLE(logger(), *throttle_clock(), 2000,
                "VO LOW confidence: inlier_ratio=%.2f → noise_scale=%.1f (VO nearly discarded)",
                vo_conf, noise_scale);
        } else if (vo_conf < 0.5) {
            double conf_scale = 1.0 + (0.5 - vo_conf) / 0.2 * 3.0;
            noise_scale = std::max(noise_scale, conf_scale);
            RCLCPP_WARN_THROTTLE(logger(), *throttle_clock(), 2000,
                "VO reduced confidence: inlier_ratio=%.2f → noise_scale=%.1f",
                vo_conf, noise_scale);
        }

        impl_->backend.addOdometryFactor(idx - 1, idx, delta, noise_scale);

        // Phase 3: preintegration 완료 시 CombinedImuFactor 추가
        if (impl_->imu_preintegrated) {
            impl_->backend.addImuFactor(idx - 1, idx, dt_sec);
            impl_->imu_preintegrated = false;
        }

        // Phase 4: ZUPT — IMU가 정지 감지 시 강한 zero-velocity + identity 제약
        if (impl_->is_zero_motion) {
            impl_->backend.addZeroVelocityConstraint(idx);
        }

        // Phase D: Reprojection factors (sliding window BA)
        if (!vo_pose.track_ids.empty() && impl_->camera_calib_set) {
            impl_->backend.addReprojectionFactors(idx, vo_pose.track_ids,
                                                   vo_pose.pixels, vo_pose.points_3d_cam);
        }
    }

    PoseOutput result = impl_->backend.optimize();
    // sanity: 발산 시 VO fallback (pos >1m 또는 rot >90°). 각도 wrap [-π,π]
    auto wrap = [](double d) {
        while (d > M_PI) d -= 2*M_PI;
        while (d < -M_PI) d += 2*M_PI;
        return d;
    };
    const double pos_diff = std::sqrt(
        (result.x - curr.x) * (result.x - curr.x) +
        (result.y - curr.y) * (result.y - curr.y) +
        (result.z - curr.z) * (result.z - curr.z));
    const double rot_diff = std::max({
        std::abs(wrap(result.roll - curr.roll)),
        std::abs(wrap(result.pitch - curr.pitch)),
        std::abs(wrap(result.yaw - curr.yaw))});
    const bool has_nan = std::isnan(result.x) || std::isnan(result.y) || std::isnan(result.z) ||
        std::isnan(result.roll) || std::isnan(result.pitch) || std::isnan(result.yaw);
    if (has_nan || pos_diff > 1.0 || rot_diff > 1.57) {
        RCLCPP_WARN_THROTTLE(logger(), *throttle_clock(), 2000,
            "factor_graph fallback to VO: %s pos_diff=%.3f rot_diff=%.3f rad (poses=%zu)",
            has_nan ? "NaN" : "diverged", pos_diff, rot_diff, impl_->backend.getNextIndex());
        result = curr;
        // 심각한 발산 시 factor graph 리셋 (velocity 폭주 방지)
        if (pos_diff > 5.0 || has_nan) {
            RCLCPP_WARN(logger(), "factor_graph RESET: severe divergence (pos_diff=%.1f)", pos_diff);
            impl_->backend.reset();
            impl_->has_prev = false;
            impl_->prev_pose = curr;
            return result;
        }
    } else {
        RCLCPP_INFO_THROTTLE(logger(), *throttle_clock(), 5000,
            "factor_graph OK: poses=%zu pos_diff=%.4f rot_diff=%.4f rad",
            impl_->backend.getNextIndex(), pos_diff, rot_diff);
    }
    impl_->prev_pose = result;
    impl_->has_prev = true;
    return result;
}

PoseOutput ImuFusionFactorGraph::fuse(const PoseInput& vo_pose, const ImuData& imu,
                                      double dt_sec,
                                      const std::vector<ImuData>& imu_samples) {
    // Phase 3: preintegrate IMU 샘플 → fuse()에서 CombinedImuFactor로 그래프에 추가
    if (!imu_samples.empty() && impl_->backend.isImuReady()) {
        impl_->backend.preintegrateImu(imu_samples);
        impl_->backend.logPreintegration();
        impl_->imu_preintegrated = true;
    }
    // Phase 4: IMU 기반 정지 감지 (ZUPT)
    impl_->is_zero_motion = FactorGraphBackend::detectZeroMotion(imu_samples);
    return fuse(vo_pose, imu, dt_sec);
}

ImuPrediction ImuFusionFactorGraph::predictFromImu(
    const std::vector<ImuData>& body_frame_samples) {
    return impl_->backend.predictFromImu(body_frame_samples);
}

void ImuFusionFactorGraph::setCameraCalibration(double fx, double fy, double cx, double cy) {
    impl_->backend.setCameraCalibration(fx, fy, cx, cy);
    impl_->camera_calib_set = true;
    RCLCPP_INFO(logger(), "Phase D: camera calibration set (fx=%.1f fy=%.1f cx=%.1f cy=%.1f)",
        fx, fy, cx, cy);
}

void ImuFusionFactorGraph::reset() {
    impl_->backend.reset();
    impl_->has_prev = false;
}

}  // namespace vo
