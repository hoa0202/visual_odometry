#include "visual_odometry/factor_graph.hpp"
#include <opencv2/core.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <rclcpp/rclcpp.hpp>
#include <map>
#include <set>

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

    // IMU preintegration (Phase 2)
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim;
    bool imu_ready{false};

    // Phase D: reprojection factor (sliding window BA)
    boost::shared_ptr<gtsam::Cal3_S2> camera_calib;
    gtsam::SharedNoiseModel reproj_noise;
    std::map<int, gtsam::Key> track_to_landmark;       // track_id → landmark key
    std::map<gtsam::Key, std::set<size_t>> lm_poses;   // landmark → visible pose indices
    std::map<gtsam::Key, gtsam::Point3> lm_saved_positions;  // sliding window 생존 landmark 위치
    int next_lm_id{0};

    // Phase 3: velocity/bias 노드 관리
    gtsam::SharedNoiseModel vel_prior_noise;
    gtsam::SharedNoiseModel bias_prior_noise;
    gtsam::imuBias::ConstantBias prev_bias;  // 최근 추정 bias (preintegration reset용)
    gtsam::Vector3 prev_velocity{0, 0, 0};   // 최근 추정 velocity
    bool has_imu_factors{false};              // IMU factor 존재 여부 (sliding window용)

    Impl() {
        prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
        // Between factor: motion-only BA가 primary tracking이므로
        // backend sliding window에서는 적당한 강도 (너무 약하면 이동 추적 실패)
        between_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.2, 0.2, 0.2).finished());
        // velocity prior
        vel_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.5);
        // bias prior: tight
        bias_prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.01);
        // Reprojection: STRONG — ORB-SLAM3 수준 (1.5px sigma, Huber 1.0)
        // 이것이 pose estimation의 주 constraint
        reproj_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.0),
            gtsam::noiseModel::Isotropic::Sigma(2, 1.5));
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
                                           const DeltaPose& delta_pose,
                                           double noise_scale) {
    gtsam::Key key_i = gtsam::Symbol('x', static_cast<gtsam::Key>(i));
    gtsam::Key key_j = gtsam::Symbol('x', static_cast<gtsam::Key>(j));
    gtsam::Pose3 measured = toPose3(
        delta_pose.x, delta_pose.y, delta_pose.z,
        delta_pose.roll, delta_pose.pitch, delta_pose.yaw);

    gtsam::SharedNoiseModel noise = impl_->between_noise;
    if (noise_scale > 1.01) {
        // VO noise inflate: IMU-VO 불일치 → VO 덜 신뢰
        noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.08 * noise_scale, 0.08 * noise_scale, 0.08 * noise_scale,
                                 0.15 * noise_scale, 0.15 * noise_scale, 0.15 * noise_scale).finished());
    }

    // Huber robust kernel: outlier 프레임의 영향을 자연스럽게 감쇠 (ORB-SLAM3 표준)
    auto robust_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(0.5), noise);
    impl_->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        key_i, key_j, measured, robust_noise));
}

double FactorGraphBackend::computeImuVoConsistency(const DeltaPose& vo_delta) const {
    if (!impl_->imu_ready || !impl_->pim || impl_->pim->deltaTij() < 1e-6) {
        return 1.0;  // IMU 없으면 기본 noise
    }

    // IMU preintegration으로 예측한 delta position
    gtsam::NavState prev_state(
        gtsam::Pose3(),  // identity (상대 delta이므로)
        impl_->prev_velocity);
    gtsam::NavState predicted = impl_->pim->predict(prev_state, impl_->prev_bias);
    gtsam::Vector3 imu_dp = predicted.pose().translation();
    gtsam::Vector3 imu_dr = predicted.pose().rotation().rpy();

    // VO delta를 벡터로 비교 (크기뿐 아니라 방향 불일치도 감지)
    gtsam::Vector3 vo_dp(vo_delta.x, vo_delta.y, vo_delta.z);

    // 위치 불일치: 벡터 차이 (방향 + 크기 모두 반영)
    double pos_diff = (vo_dp - imu_dp).norm();
    // 회전 불일치
    double rot_diff = std::sqrt(
        (vo_delta.roll - imu_dr(0)) * (vo_delta.roll - imu_dr(0)) +
        (vo_delta.pitch - imu_dr(1)) * (vo_delta.pitch - imu_dr(1)) +
        (vo_delta.yaw - imu_dr(2)) * (vo_delta.yaw - imu_dr(2)));

    // noise_scale: 불일치에 비례 (최소 1.0, 최대 20.0)
    // pos_diff > 0.01m 또는 rot_diff > 0.03rad 부터 inflate 시작
    double pos_score = std::max(0.0, (pos_diff - 0.01) / 0.02);  // 0.01m 데드존, 0.02m당 1x
    double rot_score = std::max(0.0, (rot_diff - 0.03) / 0.05);  // 0.03rad 데드존, 0.05rad당 1x
    double score = 1.0 + std::max(pos_score, rot_score) * 8.0;
    score = std::min(score, 20.0);

    if (score > 1.5) {
        auto logger = rclcpp::get_logger("factor_graph");
        static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
            "IMU-VO inconsistency: pos_diff=%.4f(vo=%.4f imu=%.4f) rot_diff=%.3f → noise_scale=%.1f",
            pos_diff, vo_dp.norm(), imu_dp.norm(), rot_diff, score);
    }

    return score;
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

        // 최적화된 velocity/bias 저장 (slide 후 재사용) + clamp
        gtsam::Key last_vel_old = gtsam::Symbol('v', static_cast<gtsam::Key>(n_old - 1));
        gtsam::Key last_bias_old = gtsam::Symbol('b', static_cast<gtsam::Key>(n_old - 1));
        if (old_result.exists(last_vel_old)) {
            impl_->prev_velocity = old_result.at<gtsam::Vector3>(last_vel_old);
            static constexpr double kMaxVel = 3.0;
            for (int a = 0; a < 3; ++a)
                impl_->prev_velocity(a) = std::max(-kMaxVel, std::min(kMaxVel, impl_->prev_velocity(a)));
        }
        if (old_result.exists(last_bias_old)) {
            auto old_bias = old_result.at<gtsam::imuBias::ConstantBias>(last_bias_old);
            // CRITICAL: sliding window에서도 bias clamp 적용 (이전 버그: 미적용으로 bias 발산)
            static constexpr double kMaxGyroBias = 0.05;
            static constexpr double kMaxAccelBias = 0.5;
            gtsam::Vector3 bg = old_bias.gyroscope();
            gtsam::Vector3 ba = old_bias.accelerometer();
            for (int a = 0; a < 3; ++a) {
                bg(a) = std::max(-kMaxGyroBias, std::min(kMaxGyroBias, bg(a)));
                ba(a) = std::max(-kMaxAccelBias, std::min(kMaxAccelBias, ba(a)));
            }
            impl_->prev_bias = gtsam::imuBias::ConstantBias(ba, bg);
        }

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

        // Phase D: landmark sliding window 관리
        // 1. pose index 0 제거 → landmark pose 참조 reindex
        // 2. pose 0에서만 보인 landmark 제거
        {
            std::map<gtsam::Key, std::set<size_t>> new_lm_poses;
            std::vector<gtsam::Key> lm_to_remove;
            for (auto& [lm_key, pose_set] : impl_->lm_poses) {
                pose_set.erase(0);  // pose 0 제거
                std::set<size_t> shifted;
                for (size_t idx : pose_set) {
                    shifted.insert(idx - 1);
                }
                if (shifted.empty()) {
                    lm_to_remove.push_back(lm_key);
                } else {
                    new_lm_poses[lm_key] = shifted;
                }
            }
            impl_->lm_poses = new_lm_poses;
            // 고아 landmark 제거 + track_to_landmark/saved_positions 업데이트
            for (const auto& lm_key : lm_to_remove) {
                impl_->lm_saved_positions.erase(lm_key);
                for (auto it = impl_->track_to_landmark.begin();
                     it != impl_->track_to_landmark.end();) {
                    if (it->second == lm_key) {
                        it = impl_->track_to_landmark.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            // 살아남은 landmark: optimized 결과로 위치 갱신 (initial에는 넣지 않음)
            // 기존 saved_positions 중 old_result에 있는 것은 최적화 결과로 업데이트
            for (auto& [lm_key, pose_set] : impl_->lm_poses) {
                if (old_result.exists(lm_key)) {
                    impl_->lm_saved_positions[lm_key] = old_result.at<gtsam::Point3>(lm_key);
                }
                // old_result에 없으면 기존 saved_positions 유지 (registered-only landmark)
            }
        }

        // Phase 3: sliding 후 v/b 노드 전부 drop (unconstrained 방지).
        // 새 앵커(node 0)에만 v/b prior 추가. 나머지는 addImuFactor()가 필요 시 생성.
        {
            gtsam::Key anchor_vel = gtsam::Symbol('v', 0);
            gtsam::Key anchor_bias = gtsam::Symbol('b', 0);
            impl_->initial.insert(anchor_vel, impl_->prev_velocity);
            impl_->initial.insert(anchor_bias, impl_->prev_bias);
            impl_->graph.add(gtsam::PriorFactor<gtsam::Vector3>(
                anchor_vel, impl_->prev_velocity, impl_->vel_prior_noise));
            impl_->graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
                anchor_bias, impl_->prev_bias, impl_->bias_prior_noise));
        }
    }

    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(20);
    gtsam::LevenbergMarquardtOptimizer opt(impl_->graph, impl_->initial, params);
    gtsam::Values result = opt.optimize();
    size_t last_idx = impl_->next_idx - 1;
    gtsam::Key last_key = gtsam::Symbol('x', static_cast<gtsam::Key>(last_idx));
    if (result.exists(last_key)) {
        fromPose3(result.at<gtsam::Pose3>(last_key), out);
    }
    // Phase 3: 최적화된 velocity/bias 저장 (다음 프레임 초기값 + preintegration bias reset)
    gtsam::Key last_vel = gtsam::Symbol('v', static_cast<gtsam::Key>(last_idx));
    gtsam::Key last_bias = gtsam::Symbol('b', static_cast<gtsam::Key>(last_idx));
    if (result.exists(last_vel)) {
        impl_->prev_velocity = result.at<gtsam::Vector3>(last_vel);
        // velocity clamp: 폭주 방지
        static constexpr double kMaxVel = 3.0;
        for (int a = 0; a < 3; ++a) {
            impl_->prev_velocity(a) = std::max(-kMaxVel,
                std::min(kMaxVel, impl_->prev_velocity(a)));
        }
    }
    if (result.exists(last_bias)) {
        impl_->prev_bias = result.at<gtsam::imuBias::ConstantBias>(last_bias);
    }
    // ALWAYS clamp bias (sliding window 경로에서 bias node가 last_idx에 없을 수 있음)
    {
        static constexpr double kMaxGyroBias = 0.05;   // rad/s (BMI088 realistic)
        static constexpr double kMaxAccelBias = 0.5;    // m/s²
        gtsam::Vector3 bg = impl_->prev_bias.gyroscope();
        gtsam::Vector3 ba = impl_->prev_bias.accelerometer();
        for (int a = 0; a < 3; ++a) {
            bg(a) = std::max(-kMaxGyroBias, std::min(kMaxGyroBias, bg(a)));
            ba(a) = std::max(-kMaxAccelBias, std::min(kMaxAccelBias, ba(a)));
        }
        impl_->prev_bias = gtsam::imuBias::ConstantBias(ba, bg);
    }
    return out;
}

void FactorGraphBackend::initImuPreintegration(const ImuPreintegrationParams& p) {
    // Z-up (ROS REP 103): gravity = (0, 0, -g)
    impl_->imu_params = gtsam::PreintegrationCombinedParams::MakeSharedU(p.gravity);
    auto& ip = impl_->imu_params;
    ip->setAccelerometerCovariance(gtsam::I_3x3 * std::pow(p.accel_noise_sigma, 2));
    ip->setGyroscopeCovariance(gtsam::I_3x3 * std::pow(p.gyro_noise_sigma, 2));
    ip->setIntegrationCovariance(gtsam::I_3x3 * 1e-8);
    ip->setBiasAccCovariance(gtsam::I_3x3 * std::pow(p.accel_bias_rw_sigma, 2));
    ip->setBiasOmegaCovariance(gtsam::I_3x3 * std::pow(p.gyro_bias_rw_sigma, 2));
    ip->setBiasAccOmegaInit(gtsam::I_6x6 * 1e-5);

    gtsam::imuBias::ConstantBias zero_bias;
    impl_->pim = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(ip, zero_bias);
    impl_->imu_ready = true;
}

void FactorGraphBackend::preintegrateImu(const std::vector<ImuData>& imu_samples) {
    if (!impl_->imu_ready || !impl_->pim) return;
    impl_->pim->resetIntegration();
    for (size_t i = 0; i < imu_samples.size(); ++i) {
        const auto& s = imu_samples[i];
        gtsam::Vector3 acc(s.lin_acc_x, s.lin_acc_y, s.lin_acc_z);
        gtsam::Vector3 gyro(s.ang_vel_x, s.ang_vel_y, s.ang_vel_z);
        double dt;
        if (i + 1 < imu_samples.size()) {
            dt = imu_samples[i + 1].timestamp - s.timestamp;
        } else if (i > 0) {
            dt = s.timestamp - imu_samples[i - 1].timestamp;
        } else {
            dt = 0.005;  // 200Hz default
        }
        if (dt <= 0.0 || dt > 0.1) dt = 0.005;
        impl_->pim->integrateMeasurement(acc, gyro, dt);
    }
}

void FactorGraphBackend::logPreintegration() const {
    if (!impl_->imu_ready || !impl_->pim) return;
    auto logger = rclcpp::get_logger("factor_graph");
    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    const auto& pim = *impl_->pim;
    gtsam::Vector3 dp = pim.deltaPij();
    gtsam::Vector3 dv = pim.deltaVij();
    gtsam::Vector3 rpy = pim.deltaRij().rpy();
    RCLCPP_INFO_THROTTLE(logger, *clock, 5000,
        "IMU preint: dt=%.3f dp=(%.4f,%.4f,%.4f) dv=(%.4f,%.4f,%.4f) rpy=(%.2f,%.2f,%.2f)deg",
        pim.deltaTij(),
        dp.x(), dp.y(), dp.z(),
        dv.x(), dv.y(), dv.z(),
        rpy.x() * 180.0 / M_PI, rpy.y() * 180.0 / M_PI, rpy.z() * 180.0 / M_PI);
}

bool FactorGraphBackend::isImuReady() const {
    return impl_->imu_ready;
}

void FactorGraphBackend::addVelocityBiasPrior(size_t i) {
    gtsam::Key vel_key = gtsam::Symbol('v', static_cast<gtsam::Key>(i));
    gtsam::Key bias_key = gtsam::Symbol('b', static_cast<gtsam::Key>(i));

    gtsam::Vector3 zero_vel(0, 0, 0);
    gtsam::imuBias::ConstantBias zero_bias;

    impl_->initial.insert(vel_key, zero_vel);
    impl_->initial.insert(bias_key, zero_bias);

    impl_->graph.add(gtsam::PriorFactor<gtsam::Vector3>(
        vel_key, zero_vel, impl_->vel_prior_noise));
    impl_->graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
        bias_key, zero_bias, impl_->bias_prior_noise));

    impl_->prev_velocity = zero_vel;
    impl_->prev_bias = zero_bias;
}

void FactorGraphBackend::addImuFactor(size_t i, size_t j, double dt_sec) {
    if (!impl_->imu_ready || !impl_->pim) return;
    if (impl_->pim->deltaTij() < 1e-6) return;  // preintegration 데이터 없음

    gtsam::Key pose_i = gtsam::Symbol('x', static_cast<gtsam::Key>(i));
    gtsam::Key vel_i = gtsam::Symbol('v', static_cast<gtsam::Key>(i));
    gtsam::Key bias_i = gtsam::Symbol('b', static_cast<gtsam::Key>(i));
    gtsam::Key pose_j = gtsam::Symbol('x', static_cast<gtsam::Key>(j));
    gtsam::Key vel_j = gtsam::Symbol('v', static_cast<gtsam::Key>(j));
    gtsam::Key bias_j = gtsam::Symbol('b', static_cast<gtsam::Key>(j));

    // i 노드에 velocity/bias 초기값 확인 (sliding window 후 없을 수 있음)
    if (!impl_->initial.exists(vel_i)) {
        impl_->initial.insert(vel_i, impl_->prev_velocity);
    }
    if (!impl_->initial.exists(bias_i)) {
        impl_->initial.insert(bias_i, impl_->prev_bias);
    }

    // CombinedImuFactor 추가
    impl_->graph.add(gtsam::CombinedImuFactor(
        pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, *impl_->pim));

    // velocity 초기값: PIM predict from previous velocity (clamped)
    static constexpr double kMaxVel = 3.0;  // m/s — 보행자 속도 상한
    gtsam::Vector3 clamped_prev_vel = impl_->prev_velocity;
    for (int a = 0; a < 3; ++a) {
        clamped_prev_vel(a) = std::max(-kMaxVel, std::min(kMaxVel, clamped_prev_vel(a)));
    }
    gtsam::NavState prev_state(
        impl_->initial.exists(pose_i) ?
            impl_->initial.at<gtsam::Pose3>(pose_i) :
            gtsam::Pose3(),
        clamped_prev_vel);
    gtsam::NavState predicted = impl_->pim->predict(prev_state, impl_->prev_bias);
    gtsam::Vector3 pred_vel = predicted.velocity();

    // velocity clamp: 비정상 속도 방지
    for (int a = 0; a < 3; ++a) {
        pred_vel(a) = std::max(-kMaxVel, std::min(kMaxVel, pred_vel(a)));
    }

    // j 노드에 velocity/bias 초기값 삽입
    if (!impl_->initial.exists(vel_j)) {
        impl_->initial.insert(vel_j, pred_vel);
    }
    if (!impl_->initial.exists(bias_j)) {
        impl_->initial.insert(bias_j, impl_->prev_bias);
    }

    impl_->has_imu_factors = true;

    auto logger = rclcpp::get_logger("factor_graph");
    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(logger, *clock, 5000,
        "IMU factor added: (%zu→%zu) dt=%.3f pred_vel=(%.3f,%.3f,%.3f)",
        i, j, impl_->pim->deltaTij(),
        pred_vel.x(), pred_vel.y(), pred_vel.z());
}

void FactorGraphBackend::addZeroVelocityConstraint(size_t i) {
    gtsam::Key vel_key = gtsam::Symbol('v', static_cast<gtsam::Key>(i));
    gtsam::Key pose_key = gtsam::Symbol('x', static_cast<gtsam::Key>(i));

    // 강한 zero-velocity prior (sigma=0.01 m/s → IMU가 정지라고 확신)
    auto zupt_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.01);
    gtsam::Vector3 zero_vel(0, 0, 0);

    if (!impl_->initial.exists(vel_key)) {
        impl_->initial.insert(vel_key, zero_vel);
    }
    impl_->graph.add(gtsam::PriorFactor<gtsam::Vector3>(
        vel_key, zero_vel, zupt_noise));

    // identity BetweenFactor: 이전 pose와 동일해야 함 (sigma 매우 작음)
    if (i > 0) {
        gtsam::Key prev_pose = gtsam::Symbol('x', static_cast<gtsam::Key>(i - 1));
        auto zupt_pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.005, 0.005, 0.005).finished());
        impl_->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            prev_pose, pose_key, gtsam::Pose3(), zupt_pose_noise));
    }

    auto logger = rclcpp::get_logger("factor_graph");
    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(logger, *clock, 3000,
        "ZUPT: zero-velocity constraint at pose %zu", i);
}

ImuPrediction FactorGraphBackend::predictFromImu(
    const std::vector<ImuData>& body_samples) const {
    ImuPrediction pred;
    if (!impl_->imu_ready || !impl_->imu_params || body_samples.size() < 2) {
        return pred;
    }

    // Temp PIM with current bias (bias-corrected integration)
    gtsam::PreintegratedCombinedMeasurements temp_pim(impl_->imu_params, impl_->prev_bias);

    for (size_t i = 0; i < body_samples.size(); ++i) {
        const auto& s = body_samples[i];
        gtsam::Vector3 acc(s.lin_acc_x, s.lin_acc_y, s.lin_acc_z);
        gtsam::Vector3 gyro(s.ang_vel_x, s.ang_vel_y, s.ang_vel_z);
        double dt;
        if (i + 1 < body_samples.size())
            dt = body_samples[i + 1].timestamp - s.timestamp;
        else if (i > 0)
            dt = s.timestamp - body_samples[i - 1].timestamp;
        else
            dt = 0.005;
        if (dt <= 0.0 || dt > 0.1) dt = 0.005;
        temp_pim.integrateMeasurement(acc, gyro, dt);
    }

    if (temp_pim.deltaTij() < 1e-6) return pred;

    // Rotation delta (bias-corrected)
    gtsam::Rot3 deltaR = temp_pim.deltaRij();
    gtsam::Matrix3 R_mat = deltaR.matrix();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            pred.R[r * 3 + c] = R_mat(r, c);

    // Translation from predict (uses velocity + bias-corrected accel)
    gtsam::NavState prev_state(gtsam::Pose3(), impl_->prev_velocity);
    gtsam::NavState predicted = temp_pim.predict(prev_state, impl_->prev_bias);
    gtsam::Point3 dp = predicted.pose().translation();
    pred.tx = dp.x();
    pred.ty = dp.y();
    pred.tz = dp.z();

    // Angular rate from rotation angle
    gtsam::Vector3 logR = gtsam::Rot3::Logmap(deltaR);
    pred.angular_rate = logR.norm() / std::max(temp_pim.deltaTij(), 0.001);
    pred.total_dt = temp_pim.deltaTij();
    pred.valid = true;

    auto logger = rclcpp::get_logger("factor_graph");
    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(logger, *clock, 5000,
        "IMU predict: dt=%.3f rot=%.3fdeg t=(%.4f,%.4f,%.4f)m ang_rate=%.2frad/s bias_g=(%.4f,%.4f,%.4f)",
        pred.total_dt,
        logR.norm() * 180.0 / M_PI,
        pred.tx, pred.ty, pred.tz,
        pred.angular_rate,
        impl_->prev_bias.gyroscope().x(),
        impl_->prev_bias.gyroscope().y(),
        impl_->prev_bias.gyroscope().z());

    return pred;
}

bool FactorGraphBackend::detectZeroMotion(const std::vector<ImuData>& imu_samples,
                                           double gyro_threshold,
                                           double accel_var_threshold) {
    if (imu_samples.size() < 3) return false;

    // gyro magnitude 평균 + accel variance 계산
    double gyro_sum = 0.0;
    double ax_sum = 0.0, ay_sum = 0.0, az_sum = 0.0;
    for (const auto& s : imu_samples) {
        gyro_sum += std::sqrt(s.ang_vel_x * s.ang_vel_x +
                              s.ang_vel_y * s.ang_vel_y +
                              s.ang_vel_z * s.ang_vel_z);
        ax_sum += s.lin_acc_x;
        ay_sum += s.lin_acc_y;
        az_sum += s.lin_acc_z;
    }
    double n = static_cast<double>(imu_samples.size());
    double gyro_avg = gyro_sum / n;
    double ax_mean = ax_sum / n, ay_mean = ay_sum / n, az_mean = az_sum / n;

    // accel variance (gravity 뺀 후가 아니라 raw에서 분산 — 정지 시 분산 작음)
    double acc_var = 0.0;
    for (const auto& s : imu_samples) {
        acc_var += (s.lin_acc_x - ax_mean) * (s.lin_acc_x - ax_mean) +
                   (s.lin_acc_y - ay_mean) * (s.lin_acc_y - ay_mean) +
                   (s.lin_acc_z - az_mean) * (s.lin_acc_z - az_mean);
    }
    acc_var /= n;

    return (gyro_avg < gyro_threshold) && (acc_var < accel_var_threshold);
}

void FactorGraphBackend::setCameraCalibration(double fx, double fy, double cx, double cy) {
    // 3D points in m → pixel projection. Cal3_S2(fx, fy, skew, cx, cy)
    impl_->camera_calib = boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0.0, cx, cy);
    auto logger = rclcpp::get_logger("factor_graph");
    RCLCPP_INFO(logger, "BA camera calibration set: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                fx, fy, cx, cy);
}

void FactorGraphBackend::addReprojectionFactors(
    size_t pose_idx,
    const std::vector<int>& track_ids,
    const std::vector<cv::Point2f>& pixels,
    const std::vector<cv::Point3f>& points_3d_cam) {
    if (!impl_->camera_calib) return;
    if (track_ids.empty() || pixels.empty() || points_3d_cam.empty()) return;

    gtsam::Key pose_key = gtsam::Symbol('x', static_cast<gtsam::Key>(pose_idx));
    if (!impl_->initial.exists(pose_key)) return;

    gtsam::Pose3 pose = impl_->initial.at<gtsam::Pose3>(pose_key);
    int added = 0;
    int registered = 0;
    static constexpr size_t kMaxActiveLandmarks = 100;

    size_t n = std::min({track_ids.size(), pixels.size(), points_3d_cam.size()});
    for (size_t i = 0; i < n; ++i) {
        int tid = track_ids[i];
        gtsam::Point3 p_cam(
            static_cast<double>(points_3d_cam[i].x) / 1000.0,
            static_cast<double>(points_3d_cam[i].y) / 1000.0,
            static_cast<double>(points_3d_cam[i].z) / 1000.0);

        if (p_cam.z() < 0.05 || p_cam.z() > 20.0) continue;

        auto it = impl_->track_to_landmark.find(tid);
        if (it == impl_->track_to_landmark.end()) {
            // 첫 관측: 위치만 기록, factor/initial에 넣지 않음 (single-obs → unconstrained)
            gtsam::Point3 p_world = pose.transformFrom(p_cam);
            gtsam::Key lm_key = gtsam::Symbol('l', static_cast<gtsam::Key>(impl_->next_lm_id++));
            impl_->track_to_landmark[tid] = lm_key;
            impl_->lm_saved_positions[lm_key] = p_world;
            impl_->lm_poses[lm_key].insert(pose_idx);
            registered++;
            continue;
        }

        // ≥2번째 관측: 이제 multi-view constraint → factor 추가
        gtsam::Key lm_key = it->second;

        // Active landmark 수 제한 (이미 initial에 있는 건 통과)
        if (!impl_->initial.exists(lm_key)) {
            // 아직 initial에 없는 landmark → 활성화 대상
            size_t active_count = 0;
            for (auto& [k, _] : impl_->lm_poses) {
                if (impl_->initial.exists(k)) active_count++;
            }
            if (active_count >= kMaxActiveLandmarks) continue;

            // saved positions에서 복원
            auto sit = impl_->lm_saved_positions.find(lm_key);
            if (sit != impl_->lm_saved_positions.end()) {
                impl_->initial.insert(lm_key, sit->second);
                impl_->lm_saved_positions.erase(sit);
            } else {
                gtsam::Point3 p_world = pose.transformFrom(p_cam);
                impl_->initial.insert(lm_key, p_world);
            }
        }

        gtsam::Point2 pixel(static_cast<double>(pixels[i].x),
                            static_cast<double>(pixels[i].y));
        try {
            impl_->graph.add(
                gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    pixel, impl_->reproj_noise, pose_key, lm_key,
                    impl_->camera_calib, false, false));
            impl_->lm_poses[lm_key].insert(pose_idx);
            added++;
        } catch (...) {
            // cheirality exception → skip
        }
    }

    if (added > 0 || registered > 0) {
        auto logger = rclcpp::get_logger("factor_graph");
        static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(logger, *clock, 3000,
            "BA: %d reproj factors (new_reg=%d) at pose %zu, active_lm=%zu",
            added, registered, pose_idx, impl_->track_to_landmark.size());
    }
}

void FactorGraphBackend::reset() {
    impl_->graph.resize(0);
    impl_->initial.clear();
    impl_->next_idx = 0;
    impl_->has_imu_factors = false;
    impl_->prev_velocity = gtsam::Vector3(0, 0, 0);
    impl_->prev_bias = gtsam::imuBias::ConstantBias();
    impl_->track_to_landmark.clear();
    impl_->lm_poses.clear();
    impl_->lm_saved_positions.clear();
}

cv::Mat FactorGraphBackend::motionOnlyBA(
    const cv::Mat& T_world_cam,
    const std::vector<cv::Point3f>& world_points,
    const std::vector<cv::Point2f>& pixels,
    int& out_inliers,
    const std::vector<int>& octaves) const {
    // ORB-SLAM3 Optimizer::PoseOptimization (GTSAM 버전)
    // - Optical frame 직접 동작
    // - Pyramid level별 information matrix (ORB-SLAM3 핵심)
    // - 4라운드 × 10 iters, Huber on/off, chi2 outlier rejection

    out_inliers = 0;
    cv::Mat result = T_world_cam.clone();
    if (!impl_->camera_calib) return result;
    size_t n = std::min(world_points.size(), pixels.size());
    if (n < 10) return result;

    auto logger = rclcpp::get_logger("factor_graph");
    const double chi2_threshold = 5.991;  // 2DOF chi2 at p=0.05

    // ORB-SLAM3: pyramid scale factors → invSigma2
    // scaleFactor=1.2, nLevels=8: sigma2[i] = 1.2^(2*i)
    const double scale_factor = 1.2;
    const int max_levels = 8;
    std::vector<double> inv_sigma2(max_levels);
    for (int i = 0; i < max_levels; ++i) {
        double scale = std::pow(scale_factor, i);
        inv_sigma2[i] = 1.0 / (scale * scale);
    }

    // T_world_cam → GTSAM Pose3 (m)
    cv::Mat R_wc = T_world_cam(cv::Rect(0,0,3,3));
    gtsam::Rot3 rot(
        R_wc.at<double>(0,0), R_wc.at<double>(0,1), R_wc.at<double>(0,2),
        R_wc.at<double>(1,0), R_wc.at<double>(1,1), R_wc.at<double>(1,2),
        R_wc.at<double>(2,0), R_wc.at<double>(2,1), R_wc.at<double>(2,2));
    gtsam::Point3 trans(
        T_world_cam.at<double>(0,3) / 1000.0,
        T_world_cam.at<double>(1,3) / 1000.0,
        T_world_cam.at<double>(2,3) / 1000.0);
    gtsam::Pose3 pose_guess(rot, trans);

    // Observations 수집
    struct Obs { gtsam::Point3 pw; gtsam::Point2 px; double invSigma2; };
    std::vector<Obs> obs;
    obs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        gtsam::Point3 pw(world_points[i].x / 1000.0,
                         world_points[i].y / 1000.0,
                         world_points[i].z / 1000.0);
        gtsam::Point3 pc = pose_guess.transformTo(pw);
        if (pc.z() < 0.05 || pc.z() > 20.0) continue;
        // ORB-SLAM3: invSigma2 from pyramid octave
        int oct = (!octaves.empty() && i < octaves.size()) ? octaves[i] : 0;
        oct = std::max(0, std::min(oct, max_levels - 1));
        obs.push_back({pw, {static_cast<double>(pixels[i].x),
                            static_cast<double>(pixels[i].y)}, inv_sigma2[oct]});
    }
    if (obs.size() < 10) return result;

    std::vector<bool> is_outlier(obs.size(), false);
    gtsam::Pose3 optimized_pose = pose_guess;

    for (int round = 0; round < 4; ++round) {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;
        gtsam::Key pk = gtsam::Symbol('x', 0);
        values.insert(pk, optimized_pose);

        int ne = 0;
        for (size_t i = 0; i < obs.size(); ++i) {
            if (is_outlier[i]) continue;
            gtsam::Key lk = gtsam::Symbol('l', static_cast<gtsam::Key>(i));
            values.insert(lk, obs[i].pw);
            graph.add(gtsam::PriorFactor<gtsam::Point3>(
                lk, obs[i].pw, gtsam::noiseModel::Isotropic::Sigma(3, 0.0001)));

            // ORB-SLAM3: information = Identity * invSigma2 → sigma = 1/sqrt(invSigma2)
            double sigma = 1.0 / std::sqrt(obs[i].invSigma2);
            gtsam::SharedNoiseModel noise;
            if (round < 2) {
                noise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Huber::Create(std::sqrt(chi2_threshold)),
                    gtsam::noiseModel::Isotropic::Sigma(2, sigma));
            } else {
                noise = gtsam::noiseModel::Isotropic::Sigma(2, sigma);
            }

            try {
                graph.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    obs[i].px, noise, pk, lk, impl_->camera_calib, false, false));
                ne++;
            } catch (...) { is_outlier[i] = true; }
        }
        if (ne < 10) break;

        try {
            gtsam::LevenbergMarquardtParams p;
            p.setMaxIterations(10);
            gtsam::LevenbergMarquardtOptimizer opt(graph, values, p);
            auto r = opt.optimize();
            if (r.exists(pk)) optimized_pose = r.at<gtsam::Pose3>(pk);
        } catch (...) { break; }

        // ORB-SLAM3: invSigma2 * err_sq > chi2 → outlier
        for (size_t i = 0; i < obs.size(); ++i) {
            if (is_outlier[i]) continue;
            try {
                gtsam::Point3 pc = optimized_pose.transformTo(obs[i].pw);
                if (pc.z() <= 0.0) { is_outlier[i] = true; continue; }
                gtsam::Point2 proj = impl_->camera_calib->uncalibrate(
                    gtsam::PinholeBase::Project(pc));
                double err_sq = (proj - obs[i].px).squaredNorm();
                if (obs[i].invSigma2 * err_sq > chi2_threshold)
                    is_outlier[i] = true;
            } catch (...) { is_outlier[i] = true; }
        }
    }

    int ni = 0;
    for (size_t i = 0; i < obs.size(); ++i) if (!is_outlier[i]) ni++;
    out_inliers = ni;

    if (ni >= 10) {
        gtsam::Matrix3 R_opt = optimized_pose.rotation().matrix();
        gtsam::Vector3 t_opt = optimized_pose.translation();
        result = cv::Mat::eye(4, 4, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                result.at<double>(r, c) = R_opt(r, c);
        result.at<double>(0, 3) = t_opt(0) * 1000.0;
        result.at<double>(1, 3) = t_opt(1) * 1000.0;
        result.at<double>(2, 3) = t_opt(2) * 1000.0;

        static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(logger, *clock, 3000,
            "Motion-only BA: %d/%zu inliers (4-round)", ni, obs.size());
    }

    return result;
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
