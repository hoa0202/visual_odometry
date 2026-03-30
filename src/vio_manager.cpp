#include "visual_odometry/vio_manager.hpp"

#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/RegistrationInfo.h>

#include <opencv2/imgproc.hpp>
#include <cmath>

namespace vo {

VIOManager::VIOManager(const VIOConfig& config, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock)
    : config_(config), logger_(logger), clock_(clock)
{
    rtabmap::ParametersMap params;

    // F2M odometry
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomF2MMaxSize(), "2000"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomF2MMaxNewFeatures(), "200"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomGuessMotion(), "true"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomResetCountdown(), "0"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomImageDecimation(), "4"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomAlignWithGround(), "false"));

    // Visual features - GFTT + optical flow
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisFeatureType(), "6"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxFeatures(), "600"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMinInliers(), "15"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorType(), "0"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisEstimationType(), "1"));

    // Optical flow
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorFlowWinSize(), "21"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorFlowIterations(), "30"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorFlowEps(), "0.01"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisCorFlowMaxLevel(), "3"));

    // BA off for real-time (biggest perf bottleneck on Jetson)
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomF2MBundleAdjustment(), "0"));

    // Depth
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMinDepth(), "0.3"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kVisMaxDepth(), "8.0"));

    // IMU filtering
    if (config_.use_imu) {
        params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomFilteringStrategy(), "1"));
    }

    // Already rectified from ZED SDK
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapImagesAlreadyRectified(), "true"));

    odom_.reset(rtabmap::Odometry::create(params));

    // SLAM backend
    if (config_.loop_closure_enable) {
        rtabmap::ParametersMap slam_params;
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapDetectionRate(), "1.0"));
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemRehearsalSimilarity(), "0.6"));
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemSTMSize(), std::to_string(config_.stm_size)));
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapMemoryThr(), std::to_string(config_.graph_window_size)));
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRtabmapImagesAlreadyRectified(), "true"));
        slam_params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOptimizerStrategy(), "2")); // GTSAM

        rtabmap_ = std::make_unique<rtabmap::Rtabmap>();
        rtabmap_->init(slam_params);
        slam_enabled_ = true;
    }

    imu_local_transform_ = rtabmap::Transform::getIdentity();

    RCLCPP_INFO(logger_, "VIOManager: rtabmap F2M odometry + %s backend, imu=%s",
        slam_enabled_ ? "SLAM" : "odom-only",
        config_.use_imu ? "ON" : "OFF");
}

VIOManager::~VIOManager() = default;

void VIOManager::setCameraParams(const CameraParams& cam) {
    cam_ = cam;
    rtab_model_ = rtabmap::CameraModel(
        cam_.fx, cam_.fy, cam_.cx, cam_.cy,
        rtabmap::Transform::getIdentity(),
        0.0,
        cv::Size(cam_.width, cam_.height));
    RCLCPP_INFO(logger_, "Camera model set: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        cam_.width, cam_.height, cam_.fx, cam_.fy, cam_.cx, cam_.cy);
}

void VIOManager::addImuMeasurement(const ImuData& imu) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_buffer_.push_back(imu);
    if (imu_buffer_.size() > kMaxImuBuffer)
        imu_buffer_.pop_front();
}

VIOOutput VIOManager::processFrame(const cv::Mat& rgb, const cv::Mat& depth, double timestamp) {
    VIOOutput output;
    output.state = VIOState::NOT_INITIALIZED;
    output.T_world_optical = cv::Mat::eye(4, 4, CV_64F);
    output.T_world_body = cv::Mat::eye(4, 4, CV_64F);

    if (cam_.fx <= 0) return output;

    cv::Mat bgr;
    if (rgb.channels() == 4)
        cv::cvtColor(rgb, bgr, cv::COLOR_BGRA2BGR);
    else if (rgb.channels() == 3)
        bgr = rgb;
    else
        cv::cvtColor(rgb, bgr, cv::COLOR_GRAY2BGR);

    // ZED default depth = mm (float32), rtabmap expects meters
    cv::Mat depth_m;
    if (depth.type() == CV_32FC1) {
        depth.convertTo(depth_m, CV_32FC1, 0.001);
    } else {
        depth_m = depth;
    }

    frame_id_++;
    rtabmap::SensorData data(bgr, depth_m, rtab_model_, frame_id_, timestamp);

    // Feed IMU data to odometry
    if (config_.use_imu) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        for (const auto& imu : imu_buffer_) {
            if (last_frame_ts_ > 0 && imu.timestamp <= last_frame_ts_) continue;
            if (imu.timestamp > timestamp) break;

            rtabmap::IMU rtab_imu(
                cv::Vec3d(imu.ang_vel_x, imu.ang_vel_y, imu.ang_vel_z),
                cv::Mat(),
                cv::Vec3d(imu.lin_acc_x, imu.lin_acc_y, imu.lin_acc_z),
                cv::Mat(),
                imu_local_transform_);

            rtabmap::SensorData imu_data;
            imu_data.setIMU(rtab_imu);
            imu_data.setStamp(imu.timestamp);
            odom_->process(imu_data);
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().timestamp <= timestamp)
            imu_buffer_.pop_front();
    }

    rtabmap::OdometryInfo odom_info;
    rtabmap::Transform pose = odom_->process(data, &odom_info);

    // Visualization data from odom_info
    {
        std::lock_guard<std::mutex> lock(vis_mutex_);
        last_vis_ = VisualizationData();

        // F2M: all tracked word keypoints in current frame
        for (const auto& w : odom_info.words)
            last_vis_.features.emplace_back(w.second);

        // matched_cur = all matched keypoints, matched_ref = inlier subset
        // (reusing these fields for color-coded visualization)
        std::set<int> inlier_ids(odom_info.reg.inliersIDs.begin(),
                                  odom_info.reg.inliersIDs.end());
        for (const auto& w : odom_info.words) {
            last_vis_.matched_cur.push_back(w.second.pt);
            if (inlier_ids.count(w.first))
                last_vis_.matched_ref.push_back(w.second.pt);
        }

        last_vis_.inliers = odom_info.reg.inliers;
        last_vis_.matches = odom_info.reg.matches;
        last_vis_.local_map_size = odom_info.localMapSize;
    }

    if (pose.isNull()) {
        state_ = VIOState::LOST;
        output.state = VIOState::LOST;
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000, "Odometry LOST! features=%d inliers=%d",
            odom_info.features, odom_info.reg.inliers);
        return output;
    }

    state_ = VIOState::TRACKING;

    cv::Mat T_odom = cv::Mat::eye(4, 4, CV_64F);
    T_odom.at<double>(0,0) = pose.r11(); T_odom.at<double>(0,1) = pose.r12(); T_odom.at<double>(0,2) = pose.r13(); T_odom.at<double>(0,3) = pose.x();
    T_odom.at<double>(1,0) = pose.r21(); T_odom.at<double>(1,1) = pose.r22(); T_odom.at<double>(1,2) = pose.r23(); T_odom.at<double>(1,3) = pose.y();
    T_odom.at<double>(2,0) = pose.r31(); T_odom.at<double>(2,1) = pose.r32(); T_odom.at<double>(2,2) = pose.r33(); T_odom.at<double>(2,3) = pose.z();

    output.T_world_optical = T_odom.clone();

    // Optical → Body (ROS: X-fwd, Y-left, Z-up)
    static const cv::Mat R_o2b = (cv::Mat_<double>(3,3) << 0,0,1, -1,0,0, 0,-1,0);
    cv::Mat R_opt = T_odom(cv::Rect(0,0,3,3));
    cv::Mat t_opt = (cv::Mat_<double>(3,1) << T_odom.at<double>(0,3), T_odom.at<double>(1,3), T_odom.at<double>(2,3));

    cv::Mat T_body = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R_body = R_o2b * R_opt * R_o2b.t();
    R_body.copyTo(T_body(cv::Rect(0,0,3,3)));
    cv::Mat t_body = R_o2b * t_opt;
    t_body.copyTo(T_body(cv::Rect(3,0,1,3)));
    output.T_world_body = T_body;

    output.state = state_;
    output.tracking_quality = std::min(1.0, odom_info.reg.inliers / 50.0);
    output.num_tracked_features = odom_info.features;
    output.num_map_matches = odom_info.reg.inliers;

    // SLAM backend - skip to keep frame rate high
    // TODO: move to async thread if loop closure needed
    // if (slam_enabled_ && rtabmap_) {
    //     cv::Mat covariance = odom_info.reg.covariance.empty() ?
    //         cv::Mat::eye(6, 6, CV_64FC1) : odom_info.reg.covariance;
    //     rtabmap_->process(data, pose, covariance);
    // }

    last_frame_ts_ = timestamp;

    RCLCPP_INFO_THROTTLE(logger_, *clock_, 1000,
        "Pose: (%.3f, %.3f, %.3f)m  quality=%.2f  feat=%d  inliers=%d  matches=%d  mapSize=%d",
        t_body.at<double>(0), t_body.at<double>(1), t_body.at<double>(2),
        output.tracking_quality, odom_info.features, odom_info.reg.inliers,
        odom_info.reg.matches, odom_info.localMapSize);

    return output;
}

void VIOManager::reset() {
    if (odom_) odom_->reset();
    state_ = VIOState::NOT_INITIALIZED;
    frame_id_ = 0;
    last_frame_ts_ = 0;
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.clear();
    }
    RCLCPP_INFO(logger_, "VIOManager reset");
}

VIOManager::VisualizationData VIOManager::getVisualizationData() const {
    std::lock_guard<std::mutex> lock(vis_mutex_);
    return last_vis_;
}

}  // namespace vo
