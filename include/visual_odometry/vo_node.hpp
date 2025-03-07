#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "visual_odometry/image_processor.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/types.hpp"

namespace vo {

class VisualOdometryNode : public rclcpp::Node {
public:
    explicit VisualOdometryNode();
    ~VisualOdometryNode() = default;

private:
    // 파라미터 관련 메서드들
    void declareParameters();
    void applyCurrentParameters();  // 메서드 선언 추가
    rcl_interfaces::msg::SetParametersResult onParamChange(
        const std::vector<rclcpp::Parameter>& params);
    
    // 콜백 함수들
    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // 멤버 변수들
    ImageProcessor image_processor_;
    FeatureDetector feature_detector_;
    CameraParams camera_params_;
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    // Publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_img_pub_;
    
    // 파라미터 콜백 핸들
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // 상태 변수들
    bool camera_info_received_{false};
    cv::Mat current_frame_;
    cv::Mat previous_frame_;
    cv::Mat current_depth_;
    cv::Mat previous_depth_;
};

} // namespace vo 