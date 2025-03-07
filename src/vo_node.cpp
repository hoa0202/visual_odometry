#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

namespace vo {

VisualOdometryNode::VisualOdometryNode() 
    : Node("visual_odometry_node")
{
    // 파라미터 선언
    declareParameters();
    
    // 초기 파라미터 값 적용
    applyCurrentParameters();
    
    // 파라미터 콜백 설정
    param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&VisualOdometryNode::onParamChange, this, std::placeholders::_1));

    // Subscribers
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/rgb/image_rect_color", 10,
        std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/depth/depth_registered", 10,
        std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/rgb/camera_info", 10,
        std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));

    // Publisher
    feature_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "feature_image", 10);

    RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been initialized");
}

void VisualOdometryNode::declareParameters()
{
    // Feature Detector 파라미터
    this->declare_parameter("feature_detector.max_features", 2000);
    this->declare_parameter("feature_detector.scale_factor", 1.2);
    this->declare_parameter("feature_detector.n_levels", 8);
    this->declare_parameter("feature_detector.visualization_type", "points");  // "points" or "rich"
    
    // Image Processor 파라미터
    this->declare_parameter("image_processor.gaussian_blur_size", 5);
    this->declare_parameter("image_processor.gaussian_sigma", 1.0);
    this->declare_parameter("image_processor.enable_histogram_eq", true);
}

void VisualOdometryNode::applyCurrentParameters() {
    // 현재 파라미터 값을 가져와서 적용
    int max_features = this->get_parameter("feature_detector.max_features").as_int();
    double scale_factor = this->get_parameter("feature_detector.scale_factor").as_double();
    int n_levels = this->get_parameter("feature_detector.n_levels").as_int();
    
    RCLCPP_INFO(this->get_logger(), "Applying parameters: max_features=%d, scale_factor=%.2f, n_levels=%d",
                max_features, scale_factor, n_levels);
    
    feature_detector_.setMaxFeatures(max_features);
    feature_detector_.setScaleFactor(scale_factor);
    feature_detector_.setNLevels(n_levels);
}

rcl_interfaces::msg::SetParametersResult VisualOdometryNode::onParamChange(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& param : params) {
        if (param.get_name() == "feature_detector.max_features") {
            feature_detector_.setMaxFeatures(param.as_int());
            RCLCPP_INFO(this->get_logger(), "Updated max_features to: %d", param.as_int());
        }
        else if (param.get_name() == "feature_detector.scale_factor") {
            feature_detector_.setScaleFactor(param.as_double());
        }
        else if (param.get_name() == "feature_detector.n_levels") {
            feature_detector_.setNLevels(param.as_int());
        }
        else if (param.get_name() == "feature_detector.visualization_type") {
            std::string viz_type = param.as_string();
            int flags = static_cast<int>(
                viz_type == "rich" ? 
                cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS : 
                cv::DrawMatchesFlags::DEFAULT);
            feature_detector_.setVisualizationFlags(flags);
        }
    }

    return result;
}

void VisualOdometryNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    try {
        if (!camera_info_received_) {
            // 카메라 행렬 초기화
            camera_params_.K = cv::Mat(3, 3, CV_64F);
            camera_params_.K.at<double>(0,0) = msg->k[0];  // fx
            camera_params_.K.at<double>(1,1) = msg->k[4];  // fy
            camera_params_.K.at<double>(0,2) = msg->k[2];  // cx
            camera_params_.K.at<double>(1,2) = msg->k[5];  // cy
            camera_params_.K.at<double>(2,2) = 1.0;

            // 왜곡 계수 초기화
            camera_params_.D = cv::Mat(msg->d.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->d.size(); ++i) {
                camera_params_.D.at<double>(i) = msg->d[i];
            }

            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera parameters received");
        }
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in camera info callback: %s", e.what());
    }
}

void VisualOdometryNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!camera_info_received_) {
        return;
    }

    try {
        // BGRA8 인코딩으로 변환
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGRA8);
        
        // BGRA에서 BGR로 변환
        cv::Mat bgr_image;
        cv::cvtColor(cv_ptr->image, bgr_image, cv::COLOR_BGRA2BGR);

        // 이전 프레임 저장
        if (!current_frame_.empty()) {
            current_frame_.copyTo(previous_frame_);
        }
        
        // 현재 프레임 저장
        bgr_image.copyTo(current_frame_);

        // 이미지 처리
        ProcessedImages processed = image_processor_.process(current_frame_);
        
        // 특징점 검출
        Features features = feature_detector_.detectFeatures(processed.gray);
        
        // 결과 시각화 및 퍼블리시
        sensor_msgs::msg::Image::SharedPtr feature_msg = 
            cv_bridge::CvImage(msg->header, "bgr8", features.visualization)
            .toImageMsg();
        feature_img_pub_->publish(*feature_msg);

        RCLCPP_INFO_THROTTLE(this->get_logger(), 
                            *this->get_clock(), 
                            1000,
                            "Detected %zu features", 
                            features.keypoints.size());
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
    }
}

void VisualOdometryNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!camera_info_received_) {
        return;
    }

    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        
        if (!current_depth_.empty()) {
            current_depth_.copyTo(previous_depth_);
        }
        
        cv_ptr->image.copyTo(current_depth_);

        RCLCPP_INFO_THROTTLE(this->get_logger(), 
                            *this->get_clock(), 
                            1000,
                            "Depth image processed. Size: %dx%d", 
                            current_depth_.cols, 
                            current_depth_.rows);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
    }
}

} // namespace vo