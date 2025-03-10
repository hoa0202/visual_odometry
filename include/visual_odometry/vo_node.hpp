#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/highgui.hpp>
#include "visual_odometry/image_processor.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/types.hpp"
#include <thread>
#include <mutex>

namespace vo {

class VisualOdometryNode : public rclcpp::Node {
public:
    explicit VisualOdometryNode();
    virtual ~VisualOdometryNode();

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
    bool features_detected_{false};
    cv::Mat current_frame_;
    cv::Mat previous_frame_;
    cv::Mat current_depth_;
    cv::Mat previous_depth_;

    // 시각화 관련 변수들
    bool show_original_{true};
    bool show_features_{true};
    int window_width_{800};
    int window_height_{600};
    int window_pos_x_{100};
    int window_pos_y_{100};
    const std::string original_window_name_{"Original Image"};
    const std::string feature_window_name_{"Feature Detection Result"};

    std::thread display_thread_;
    std::mutex frame_mutex_;
    cv::Mat display_frame_;
    bool should_exit_{false};
    
    // FPS 계산을 위한 변수들
    rclcpp::Time last_fps_time_{0};
    int fps_frame_count_{0};
    double fps_total_process_time_{0.0};
    
    void displayLoop();
};

} // namespace vo 