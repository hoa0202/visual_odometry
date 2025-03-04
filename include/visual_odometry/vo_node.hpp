#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "visual_odometry/image_processor.hpp"
#include "visual_odometry/types.hpp"

namespace vo {
class VisualOdometryNode : public rclcpp::Node {
public:
    explicit VisualOdometryNode();
    ~VisualOdometryNode() = default;

private:
    // 멤버 변수
    ImageProcessor image_processor_;
    CameraParams camera_params_;
    cv::Mat current_frame_;
    cv::Mat previous_frame_;
    cv::Mat current_depth_;
    cv::Mat previous_depth_;
    bool camera_info_received_{false};
    int frame_count_{0};

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    // Callback 메서드
    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
};
} // namespace vo 