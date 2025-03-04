#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

namespace vo {

VisualOdometryNode::VisualOdometryNode() 
    : Node("visual_odometry_node")
{
    // RGB 이미지 구독
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/rgb/image_rect_color", 10,
        std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));

    // Depth 이미지 구독
    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/depth/depth_registered", 10,
        std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

    // Camera Info 구독
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/rgb/camera_info", 10,
        std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been initialized");
}

void VisualOdometryNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!camera_info_received_) {
        try {
            // Camera Matrix (K) 설정
            camera_params_.K = cv::Mat::zeros(3, 3, CV_64F);
            camera_params_.K.at<double>(0,0) = msg->k[0];  // fx
            camera_params_.K.at<double>(1,1) = msg->k[4];  // fy
            camera_params_.K.at<double>(0,2) = msg->k[2];  // cx
            camera_params_.K.at<double>(1,2) = msg->k[5];  // cy
            camera_params_.K.at<double>(2,2) = 1.0;

            // 왜곡 계수 (D) 설정
            camera_params_.D = cv::Mat(msg->d.size(), 1, CV_64F);
            for(size_t i = 0; i < msg->d.size(); i++) {
                camera_params_.D.at<double>(i) = msg->d[i];
            }

            camera_params_.fx = msg->k[0];
            camera_params_.fy = msg->k[4];
            camera_params_.cx = msg->k[2];
            camera_params_.cy = msg->k[5];

            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera parameters have been initialized");
            RCLCPP_INFO(this->get_logger(), "fx: %f, fy: %f, cx: %f, cy: %f",
                       camera_params_.fx, camera_params_.fy, 
                       camera_params_.cx, camera_params_.cy);
        }
        catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error in camera info callback: %s", e.what());
        }
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

        RCLCPP_INFO_THROTTLE(this->get_logger(), 
                            *this->get_clock(), 
                            1000,
                            "RGB image processed. Size: %dx%d", 
                            current_frame_.cols, 
                            current_frame_.rows);
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