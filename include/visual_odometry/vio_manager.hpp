#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"

#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/IMU.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Statistics.h>

#include <mutex>
#include <deque>

namespace vo {

class VIOManager {
public:
    VIOManager(const VIOConfig& config, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock);
    ~VIOManager();

    void setCameraParams(const CameraParams& cam);
    void addImuMeasurement(const ImuData& imu);

    VIOOutput processFrame(const cv::Mat& rgb, const cv::Mat& depth, double timestamp);

    VIOState getState() const { return state_; }
    void reset();

    struct VisualizationData {
        std::vector<cv::KeyPoint> features;
        std::vector<cv::Point2f> matched_cur;
        std::vector<cv::Point2f> matched_ref;
        int inliers{0};
        int matches{0};
        int local_map_size{0};
    };
    VisualizationData getVisualizationData() const;

private:
    rclcpp::Logger& vlog() { return logger_; }
    rclcpp::Clock::SharedPtr vclock() { return clock_; }

    VIOConfig config_;
    rclcpp::Logger logger_;
    rclcpp::Clock::SharedPtr clock_;
    VIOState state_{VIOState::NOT_INITIALIZED};

    CameraParams cam_;
    rtabmap::CameraModel rtab_model_;

    // RTAB-Map odometry (F2M)
    std::unique_ptr<rtabmap::Odometry> odom_;

    // RTAB-Map SLAM backend
    std::unique_ptr<rtabmap::Rtabmap> rtabmap_;
    bool slam_enabled_{false};

    // IMU buffer
    std::deque<ImuData> imu_buffer_;
    std::mutex imu_mutex_;
    static constexpr size_t kMaxImuBuffer = 2000;
    double last_frame_ts_{0};

    // Camera←IMU transform (for rtabmap IMU local_transform)
    rtabmap::Transform imu_local_transform_;

    mutable std::mutex vis_mutex_;
    VisualizationData last_vis_;

    int frame_id_{0};
};

}  // namespace vo
