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
        impl_->backend.addOdometryFactor(idx - 1, idx, delta);
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
    // Phase 2: preintegrate IMU 샘플 + 로그 (factor graph에는 아직 적용 안 함)
    if (!imu_samples.empty() && impl_->backend.isImuReady()) {
        impl_->backend.preintegrateImu(imu_samples);
        impl_->backend.logPreintegration();
    }
    // 기존 VO-only factor graph 로직
    return fuse(vo_pose, imu, dt_sec);
}

void ImuFusionFactorGraph::reset() {
    impl_->backend.reset();
    impl_->has_prev = false;
}

}  // namespace vo
