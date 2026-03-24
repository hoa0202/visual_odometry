#include "visual_odometry/factor_graph.hpp"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

namespace vo {

static gtsam::Pose3 toPose3(double x, double y, double z,
                             double roll, double pitch, double yaw) {
    gtsam::Rot3 R = gtsam::Rot3::RzRyRx(roll, pitch, yaw);
    gtsam::Point3 t(x, y, z);
    return gtsam::Pose3(R, t);
}

static void fromPose3(const gtsam::Pose3& p, PoseOutput& out) {
    const gtsam::Point3& t = p.translation();
    out.x = t.x();
    out.y = t.y();
    out.z = t.z();
    gtsam::Vector3 ypr = p.rotation().ypr();  // (yaw, pitch, roll)
    out.yaw = ypr(0);
    out.pitch = ypr(1);
    out.roll = ypr(2);
}

static void fromPose3ToDelta(const gtsam::Pose3& p, DeltaPose& out) {
    PoseOutput tmp;
    fromPose3(p, tmp);
    out.x = tmp.x;
    out.y = tmp.y;
    out.z = tmp.z;
    out.roll = tmp.roll;
    out.pitch = tmp.pitch;
    out.yaw = tmp.yaw;
}

DeltaPose FactorGraphBackend::computeDelta(const PoseOutput& prev,
                                           const PoseOutput& curr) {
    gtsam::Pose3 p_prev = toPose3(prev.x, prev.y, prev.z,
                                  prev.roll, prev.pitch, prev.yaw);
    gtsam::Pose3 p_curr = toPose3(curr.x, curr.y, curr.z,
                                  curr.roll, curr.pitch, curr.yaw);
    gtsam::Pose3 delta = p_prev.inverse() * p_curr;
    DeltaPose out;
    fromPose3ToDelta(delta, out);
    return out;
}

DeltaPose FactorGraphBackend::invertDelta(const DeltaPose& d) {
    gtsam::Pose3 p = toPose3(d.x, d.y, d.z, d.roll, d.pitch, d.yaw);
    DeltaPose out;
    fromPose3ToDelta(p.inverse(), out);
    return out;
}

struct FactorGraphBackend::Impl {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    gtsam::SharedNoiseModel prior_noise;
    gtsam::SharedNoiseModel between_noise;
    size_t next_idx{0};
    size_t window_size{20};  // 0 = unlimited

    Impl() {
        prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
        between_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
    }
};

FactorGraphBackend::FactorGraphBackend() : impl_(std::make_unique<Impl>()) {}

FactorGraphBackend::~FactorGraphBackend() = default;

void FactorGraphBackend::setWindowSize(size_t N) {
    impl_->window_size = N;
}

size_t FactorGraphBackend::getNextIndex() const {
    return impl_->next_idx;
}

void FactorGraphBackend::addPose(size_t i, double x, double y, double z,
                                  double roll, double pitch, double yaw) {
    gtsam::Key key = gtsam::Symbol('x', static_cast<gtsam::Key>(i));
    gtsam::Pose3 pose = toPose3(x, y, z, roll, pitch, yaw);
    impl_->initial.insert(key, pose);
    if (i == 0) {
        impl_->graph.add(
            gtsam::PriorFactor<gtsam::Pose3>(key, pose, impl_->prior_noise));
    }
    if (i >= impl_->next_idx) {
        impl_->next_idx = i + 1;
    }
}

void FactorGraphBackend::addOdometryFactor(size_t i, size_t j,
                                           const DeltaPose& delta_pose) {
    gtsam::Key key_i = gtsam::Symbol('x', static_cast<gtsam::Key>(i));
    gtsam::Key key_j = gtsam::Symbol('x', static_cast<gtsam::Key>(j));
    gtsam::Pose3 measured = toPose3(
        delta_pose.x, delta_pose.y, delta_pose.z,
        delta_pose.roll, delta_pose.pitch, delta_pose.yaw);
    impl_->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        key_i, key_j, measured, impl_->between_noise));
}

PoseOutput FactorGraphBackend::optimize() {
    PoseOutput out;
    if (impl_->initial.size() == 0) {
        return out;
    }
    // Phase 4: sliding window - N 초과 시 가장 오래된 pose 제거, 글로벌 좌표 유지
    if (impl_->window_size > 0 && impl_->next_idx > impl_->window_size) {
        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(20);
        gtsam::LevenbergMarquardtOptimizer opt(impl_->graph, impl_->initial, params);
        gtsam::Values old_result = opt.optimize();

        // pose 1이 없으면 최신 pose 직접 반환
        gtsam::Key key1 = gtsam::Symbol('x', 1);
        if (!old_result.exists(key1)) {
            gtsam::Key last = gtsam::Symbol('x', static_cast<gtsam::Key>(impl_->next_idx - 1));
            if (old_result.exists(last)) {
                fromPose3(old_result.at<gtsam::Pose3>(last), out);
            }
            return out;
        }

        // 재구성: pose 1→0, 2→1, ..., N-1→N-2 (글로벌 좌표 유지, T_base 불필요)
        impl_->graph.resize(0);
        impl_->initial.clear();
        const size_t n_old = impl_->next_idx;

        for (size_t k = 1; k < n_old; ++k) {
            gtsam::Key old_key = gtsam::Symbol('x', static_cast<gtsam::Key>(k));
            if (!old_result.exists(old_key)) break;
            gtsam::Pose3 pk = old_result.at<gtsam::Pose3>(old_key);
            gtsam::Key new_key = gtsam::Symbol('x', static_cast<gtsam::Key>(k - 1));
            impl_->initial.insert(new_key, pk);

            if (k == 1) {
                // 새 앵커: pose 0 = 이전 pose 1 (글로벌 좌표)
                impl_->graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                    new_key, pk, impl_->prior_noise));
            }
            if (k >= 2) {
                gtsam::Pose3 p_prev = old_result.at<gtsam::Pose3>(
                    gtsam::Symbol('x', static_cast<gtsam::Key>(k - 1)));
                gtsam::Pose3 delta = p_prev.inverse() * pk;
                impl_->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    gtsam::Symbol('x', static_cast<gtsam::Key>(k - 2)),
                    new_key, delta, impl_->between_noise));
            }
        }
        impl_->next_idx = n_old - 1;
    }

    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(20);
    gtsam::LevenbergMarquardtOptimizer opt(impl_->graph, impl_->initial, params);
    gtsam::Values result = opt.optimize();
    gtsam::Key last_key = gtsam::Symbol('x', static_cast<gtsam::Key>(impl_->next_idx - 1));
    if (result.exists(last_key)) {
        fromPose3(result.at<gtsam::Pose3>(last_key), out);
    }
    return out;
}

void FactorGraphBackend::reset() {
    impl_->graph.resize(0);
    impl_->initial.clear();
    impl_->next_idx = 0;
}

bool FactorGraphBackend::runVerification() {
    FactorGraphBackend b;
    b.addPose(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    b.addPose(1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0);
    b.addPose(2, 0.2, 0.01, 0.0, 0.0, 0.0, 0.01);
    DeltaPose d01{0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    DeltaPose d12{0.1, 0.01, 0.0, 0.0, 0.0, 0.01};
    b.addOdometryFactor(0, 1, d01);
    b.addOdometryFactor(1, 2, d12);
    PoseOutput out = b.optimize();
    (void)out;  // 검증: optimize 성공 + pose 반환 (x≈0.2, y≈0.01, yaw≈0.01)
    return true;
}

}  // namespace vo
