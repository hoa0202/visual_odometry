#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/highgui.hpp>
#include "visual_odometry/image_processor.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/types.hpp"
#include <thread>
#include <mutex>
#include "visual_odometry/zed_interface.hpp"
#include <deque>
#include <queue>
#include <condition_variable>

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
    
    // ZED 인터페이스
    std::unique_ptr<ZEDInterface> zed_interface_;
    std::string input_source_{"ros2"};
    
    // 이미지 획득 메서드
    bool getImages(cv::Mat& rgb, cv::Mat& depth);
    
    void displayLoop();

    // ZED SDK 관련
    rclcpp::TimerBase::SharedPtr zed_timer_;
    void zedTimerCallback();
    void processImages(const cv::Mat& rgb, const cv::Mat& depth);

    cv::Mat display_frame_original_;  // 원본 이미지용
    cv::Mat display_frame_features_;  // 특징점 이미지용

    // 이미지 처리를 위한 버퍼들
    cv::Mat resized_frame_;
    cv::Mat gray_buffer_;
    cv::Mat resized_original_;
    cv::Mat resized_features_;

    // 타이밍 측정을 위한 변수들
    rclcpp::Time start_time_;
    rclcpp::Time resize_start_;
    rclcpp::Time feature_start_;
    rclcpp::Time viz_start_;

    // FPS 측정을 위한 변수들
    std::deque<double> original_frame_times_;  // 원본 이미지용
    std::deque<double> feature_frame_times_;   // 특징점 검출용
    rclcpp::Time last_fps_print_time_;
    double original_fps_;
    double feature_fps_;
    int fps_window_size_;
    double zed_acquisition_time_;

    std::queue<std::pair<cv::Mat, cv::Mat>> image_queue_;
    std::mutex image_queue_mutex_;
    std::condition_variable image_ready_;
    std::thread processing_thread_;
    void processingLoop();
};

} // namespace vo 