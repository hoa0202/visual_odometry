#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include "visual_odometry/msg/vo_state.hpp"
#include "visual_odometry/types.hpp"
#include "visual_odometry/vio_manager.hpp"
#include "visual_odometry/zed_interface.hpp"
#include "visual_odometry/visualization.hpp"
#include "visual_odometry/image_processor.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/logger.hpp"
#include "visual_odometry/resource_monitor.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <deque>

namespace vo {

class VisualOdometryNode : public rclcpp::Node {
public:
    explicit VisualOdometryNode();
    virtual ~VisualOdometryNode();

private:
    // ─── Logging / Monitoring ───────────────────────────────────────────────
    std::unique_ptr<Logger> logger_;
    std::unique_ptr<ResourceMonitor> resource_monitor_;

    // ─── Parameters ─────────────────────────────────────────────────────────
    void declareParameters();
    void applyCurrentParameters();
    rcl_interfaces::msg::SetParametersResult onParamChange(
        const std::vector<rclcpp::Parameter>& params);
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // ─── VIO Manager (THE pipeline) ─────────────────────────────────────────
    std::unique_ptr<VIOManager> vio_manager_;
    CameraParams camera_params_;
    bool camera_info_received_{false};

    // ─── ROS2 I/O ───────────────────────────────────────────────────────────
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_img_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<visual_odometry::msg::VOState>::SharedPtr vo_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // ─── Callbacks ──────────────────────────────────────────────────────────
    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // ─── Image Processing ───────────────────────────────────────────────────
    void processImages(const cv::Mat& rgb, const cv::Mat& depth);
    void publishResults(const VIOOutput& output, double processing_time_ms);
    void publishStaticTransform();
    VIOConfig loadVIOConfig();

    // ─── ZED SDK ────────────────────────────────────────────────────────────
    std::unique_ptr<ZEDInterface> zed_interface_;
    std::string input_source_{"ros2"};
    rclcpp::TimerBase::SharedPtr zed_timer_;
    void zedTimerCallback();
    bool getImages(cv::Mat& rgb, cv::Mat& depth);
    std::thread imu_poll_thread_;
    void imuPollLoop();
    cv::Mat R_cam_imu_;   // rotation: IMU frame → camera (IMAGE) frame
    bool has_cam_imu_tf_{false};

    // ─── Processing Thread ──────────────────────────────────────────────────
    std::queue<std::pair<cv::Mat, cv::Mat>> image_queue_;
    std::mutex image_queue_mutex_;
    std::condition_variable image_ready_;
    std::thread processing_thread_;
    bool should_exit_{false};
    void processingLoop();

    // ─── Depth buffer (ros2 mode) ───────────────────────────────────────────
    cv::Mat current_depth_;
    std::mutex depth_mutex_;

    // ─── Visualization ──────────────────────────────────────────────────────
    std::unique_ptr<Visualizer> visualizer_;
    bool show_original_{false};
    bool show_features_{false};
    bool show_matches_{false};
    int window_width_{800};
    int window_height_{600};
    int window_pos_x_{100};
    int window_pos_y_{100};
    cv::Mat prev_frame_;

    // ─── FPS ────────────────────────────────────────────────────────────────
    int fps_window_size_{60};
    int frame_count_{0};
    double fps_start_time_{0.0};
    double current_fps_{0.0};
};

}  // namespace vo
