#include "visual_odometry/imu_fusion_ekf.hpp"
#include <cmath>
#include <algorithm>

namespace vo {

ImuFusionEKF::ImuFusionEKF(const EKFParams& params) : params_(params) {
    x_ = cv::Mat::zeros(N, 1, CV_64F);
    P_ = cv::Mat::eye(N, N, CV_64F);
    // 초기 공분산
    P_.at<double>(0, 0) = P_.at<double>(1, 1) = P_.at<double>(2, 2) = 0.01;   // position
    P_.at<double>(3, 3) = P_.at<double>(4, 4) = P_.at<double>(5, 5) = 0.1;    // velocity
    P_.at<double>(6, 6) = P_.at<double>(7, 7) = P_.at<double>(8, 8) = 0.01;  // rpy
    P_.at<double>(9, 9) = P_.at<double>(10, 10) = P_.at<double>(11, 11) = 1e-4;  // gyro bias
    P_.at<double>(12, 12) = P_.at<double>(13, 13) = P_.at<double>(14, 14) = 1e-4; // accel bias
}

cv::Mat ImuFusionEKF::eulerRateFromOmega(double roll, double pitch, double /*yaw*/,
                                         double wx, double wy, double wz) const {
    double cr = std::cos(roll), sr = std::sin(roll);
    double cp = std::cos(pitch), tp = std::tan(pitch);
    const double eps = 1e-6;
    if (std::abs(cp) < eps) cp = eps;
    if (std::abs(tp) > 1e3) tp = (tp > 0) ? 1e3 : -1e3;

    cv::Mat rate = (cv::Mat_<double>(3, 1) <<
        wx + sr * tp * wy + cr * tp * wz,
        cr * wy - sr * wz,
        sr / cp * wy + cr / cp * wz);
    return rate;
}

cv::Mat ImuFusionEKF::eulerToR(double roll, double pitch, double yaw) const {
    double cr = std::cos(roll), sr = std::sin(roll);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cy = std::cos(yaw), sy = std::sin(yaw);
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        cp*cy,  sr*sp*cy - cr*sy,  cr*sp*cy + sr*sy,
        cp*sy,  sr*sp*sy + cr*cy,  cr*sp*sy - sr*cy,
        -sp,    sr*cp,             cr*cp);
    return R;
}

void ImuFusionEKF::predict(const ImuData& imu, double dt) {
    if (dt <= 0) return;

    double px = x_.at<double>(0), py = x_.at<double>(1), pz = x_.at<double>(2);
    double vx = x_.at<double>(3), vy = x_.at<double>(4), vz = x_.at<double>(5);
    double roll = x_.at<double>(6), pitch = x_.at<double>(7), yaw = x_.at<double>(8);
    double bgx = x_.at<double>(9), bgy = x_.at<double>(10), bgz = x_.at<double>(11);
    double bax = x_.at<double>(12), bay = x_.at<double>(13), baz = x_.at<double>(14);

    double wx = imu.ang_vel_x - bgx, wy = imu.ang_vel_y - bgy, wz = imu.ang_vel_z - bgz;
    double ax = imu.lin_acc_x - bax, ay = imu.lin_acc_y - bay, az = imu.lin_acc_z - baz;

    // rpy integration
    cv::Mat rpy_rate = eulerRateFromOmega(roll, pitch, yaw, wx, wy, wz);
    roll += rpy_rate.at<double>(0, 0) * dt;
    pitch += rpy_rate.at<double>(1, 0) * dt;
    yaw += rpy_rate.at<double>(2, 0) * dt;

    // velocity: v_dot = R * (a - ba) + g, g=(0,0,-9.81)
    cv::Mat R = eulerToR(roll, pitch, yaw);
    cv::Mat acc_body = (cv::Mat_<double>(3, 1) << ax, ay, az);
    cv::Mat acc_world = R * acc_body;
    acc_world.at<double>(2, 0) -= g_;
    vx += acc_world.at<double>(0, 0) * dt;
    vy += acc_world.at<double>(1, 0) * dt;
    vz += acc_world.at<double>(2, 0) * dt;

    // position
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    x_.at<double>(0) = px; x_.at<double>(1) = py; x_.at<double>(2) = pz;
    x_.at<double>(3) = vx; x_.at<double>(4) = vy; x_.at<double>(5) = vz;
    x_.at<double>(6) = roll; x_.at<double>(7) = pitch; x_.at<double>(8) = yaw;
    // bias unchanged

    // Covariance: P = F*P*F' + Q. Simplified: F block-diagonal, Q diagonal
    cv::Mat Q = cv::Mat::eye(N, N, CV_64F);
    Q.at<double>(0, 0) = Q.at<double>(1, 1) = Q.at<double>(2, 2) = 0.001 * dt * dt;
    Q.at<double>(3, 3) = Q.at<double>(4, 4) = Q.at<double>(5, 5) = 0.01 * dt * dt;
    Q.at<double>(6, 6) = Q.at<double>(7, 7) = Q.at<double>(8, 8) = 0.001 * dt * dt;
    Q.at<double>(9, 9) = Q.at<double>(10, 10) = Q.at<double>(11, 11) = 1e-8 * dt;
    Q.at<double>(12, 12) = Q.at<double>(13, 13) = Q.at<double>(14, 14) = 1e-8 * dt;

    P_ = P_ + Q;
}

void ImuFusionEKF::predictOrientationOnly(const ImuData& imu, double dt) {
    if (dt <= 0) return;
    double roll = x_.at<double>(6), pitch = x_.at<double>(7), yaw = x_.at<double>(8);
    double bgx = x_.at<double>(9), bgy = x_.at<double>(10), bgz = x_.at<double>(11);
    double wx = imu.ang_vel_x - bgx, wy = imu.ang_vel_y - bgy, wz = imu.ang_vel_z - bgz;
    cv::Mat rpy_rate = eulerRateFromOmega(roll, pitch, yaw, wx, wy, wz);
    roll += rpy_rate.at<double>(0, 0) * dt;
    pitch += rpy_rate.at<double>(1, 0) * dt;
    yaw += rpy_rate.at<double>(2, 0) * dt;
    x_.at<double>(6) = roll;
    x_.at<double>(7) = pitch;
    x_.at<double>(8) = yaw;
    // position, velocity unchanged; zero velocity (vision loss)
    x_.at<double>(3) = x_.at<double>(4) = x_.at<double>(5) = 0.0;
}

bool ImuFusionEKF::update(const PoseInput& vo_pose) {
    // Measurement: z = [px, py, pz, roll, pitch, yaw], 6-dim
    cv::Mat z = (cv::Mat_<double>(6, 1) <<
        vo_pose.x, vo_pose.y, vo_pose.z,
        vo_pose.roll, vo_pose.pitch, vo_pose.yaw);

    // H: observe p(0:3) and rpy(6:9)
    cv::Mat H = cv::Mat::zeros(6, N, CV_64F);
    for (int i = 0; i < 3; ++i) H.at<double>(i, i) = 1.0;
    for (int i = 0; i < 3; ++i) H.at<double>(3 + i, 6 + i) = 1.0;

    cv::Mat hx = (cv::Mat_<double>(6, 1) <<
        x_.at<double>(0), x_.at<double>(1), x_.at<double>(2),
        x_.at<double>(6), x_.at<double>(7), x_.at<double>(8));

    cv::Mat y = z - hx;  // innovation

    // R: measurement noise - position R↑ → VO 덜 신뢰, 한쪽 drift 완화
    cv::Mat R = cv::Mat::eye(6, 6, CV_64F);
    R.at<double>(0, 0) = R.at<double>(1, 1) = R.at<double>(2, 2) = 0.01;   // position
    R.at<double>(3, 3) = R.at<double>(4, 4) = R.at<double>(5, 5) = 0.0005; // orientation

    cv::Mat S = H * P_ * H.t() + R;
    cv::Mat S_inv;
    cv::invert(S, S_inv, cv::DECOMP_SVD);

    // Chi-squared gating: d² = y'*S⁻¹*y, reject if > χ²(6, 0.99)
    cv::Mat d2_mat = y.t() * S_inv * y;
    double d2 = d2_mat.at<double>(0, 0);
    if (d2 > params_.chi2_threshold) return false;  // outlier, skip

    // Huber robust: clip innovation to limit outlier influence
    for (int i = 0; i < 3; ++i) {
        double yi = y.at<double>(i, 0);
        double delta = params_.huber_pos_m;
        if (std::abs(yi) > delta) y.at<double>(i, 0) = (yi > 0 ? delta : -delta);
    }
    for (int i = 3; i < 6; ++i) {
        double yi = y.at<double>(i, 0);
        double delta = params_.huber_rot_rad;
        if (std::abs(yi) > delta) y.at<double>(i, 0) = (yi > 0 ? delta : -delta);
    }

    cv::Mat Kt;
    cv::solve(S, H * P_, Kt, cv::DECOMP_SVD);
    cv::Mat K = Kt.t();

    x_ = x_ + K * y;
    cv::Mat I_KH = cv::Mat::eye(N, N, CV_64F) - K * H;
    P_ = I_KH * P_ * I_KH.t() + K * R * K.t();
    return true;
}

PoseOutput ImuFusionEKF::fuse(const PoseInput& vo_pose, const ImuData& imu, double dt_sec) {
    if (!initialized_) {
        x_.at<double>(0) = vo_pose.x;
        x_.at<double>(1) = vo_pose.y;
        x_.at<double>(2) = vo_pose.z;
        x_.at<double>(6) = vo_pose.roll;
        x_.at<double>(7) = vo_pose.pitch;
        x_.at<double>(8) = vo_pose.yaw;
        initialized_ = true;
    }

    // Stationary detection: IMU says no motion → position hold (predict 전에 판단)
    bool stationary = false;
    if (imu.valid) {
        double no = imu.ang_vel_x * imu.ang_vel_x + imu.ang_vel_y * imu.ang_vel_y + imu.ang_vel_z * imu.ang_vel_z;
        double na = imu.lin_acc_x * imu.lin_acc_x + imu.lin_acc_y * imu.lin_acc_y + imu.lin_acc_z * imu.lin_acc_z;
        stationary = (no < 0.05) && (std::abs(std::sqrt(na) - g_) < 1.2);  // |ω|<0.22 rad/s, |a|≈g±1.2
    }

    if (imu.valid) {
        if (stationary || !vo_pose.valid) {
            predictOrientationOnly(imu, dt_sec);  // 정지 또는 PnP fail → position hold
        } else {
            predict(imu, dt_sec);
        }
    }

    if (vo_pose.valid && !stationary) {
        if (update(vo_pose)) {
            // velocity: EMA with VO delta to reduce drift (직접 대입 시 한쪽으로 쭉 밀림)
            if (p_prev_valid_ && dt_sec > 0.01) {
                const double alpha = 0.4;  // 0.4*vo + 0.6*prev
                double vx_vo = (x_.at<double>(0) - p_prev_[0]) / dt_sec;
                double vy_vo = (x_.at<double>(1) - p_prev_[1]) / dt_sec;
                double vz_vo = (x_.at<double>(2) - p_prev_[2]) / dt_sec;
                x_.at<double>(3) = alpha * vx_vo + (1.0 - alpha) * x_.at<double>(3);
                x_.at<double>(4) = alpha * vy_vo + (1.0 - alpha) * x_.at<double>(4);
                x_.at<double>(5) = alpha * vz_vo + (1.0 - alpha) * x_.at<double>(5);
            }
        }
        p_prev_[0] = x_.at<double>(0);
        p_prev_[1] = x_.at<double>(1);
        p_prev_[2] = x_.at<double>(2);
        p_prev_valid_ = true;
    } else if (stationary) {
        // 카메라 정지 시 velocity=0, VO 무시 (앞에 움직이는 물체 영향 제거)
        x_.at<double>(3) = x_.at<double>(4) = x_.at<double>(5) = 0.0;
        p_prev_[0] = x_.at<double>(0);
        p_prev_[1] = x_.at<double>(1);
        p_prev_[2] = x_.at<double>(2);
        p_prev_valid_ = true;
    } else if (!vo_pose.valid) {
        // PnP fail → predictOrientationOnly 이미 적용됨, p_prev 유지
        p_prev_[0] = x_.at<double>(0);
        p_prev_[1] = x_.at<double>(1);
        p_prev_[2] = x_.at<double>(2);
        p_prev_valid_ = true;
    }

    PoseOutput out;
    out.x = x_.at<double>(0);
    out.y = x_.at<double>(1);
    out.z = x_.at<double>(2);
    out.roll = x_.at<double>(6);
    out.pitch = x_.at<double>(7);
    out.yaw = x_.at<double>(8);
    return out;
}

void ImuFusionEKF::reset() {
    x_.setTo(0);
    p_prev_valid_ = false;
    P_ = cv::Mat::eye(N, N, CV_64F);
    P_.at<double>(0, 0) = P_.at<double>(1, 1) = P_.at<double>(2, 2) = 0.01;
    P_.at<double>(3, 3) = P_.at<double>(4, 4) = P_.at<double>(5, 5) = 0.1;
    P_.at<double>(6, 6) = P_.at<double>(7, 7) = P_.at<double>(8, 8) = 0.01;
    P_.at<double>(9, 9) = P_.at<double>(10, 10) = P_.at<double>(11, 11) = 1e-4;
    P_.at<double>(12, 12) = P_.at<double>(13, 13) = P_.at<double>(14, 14) = 1e-4;
    initialized_ = false;
}

}  // namespace vo
