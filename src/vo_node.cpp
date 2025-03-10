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

    // QoS 프로파일 설정
    auto qos = rclcpp::QoS(1).best_effort().durability_volatile();

    // Subscribers
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "rgb_image", qos,
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

    // OpenCV 윈도우 생성 및 위치 설정
    try {
        if (show_original_) {
            cv::namedWindow(original_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
            cv::moveWindow(original_window_name_, window_pos_x_, window_pos_y_);
        }
        if (show_features_) {
            cv::namedWindow(feature_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
            cv::moveWindow(feature_window_name_, 
                          window_pos_x_ + window_width_ + 30,
                          window_pos_y_);
        }
    } catch (const cv::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "Failed to create windows: %s", e.what());
        show_original_ = false;
        show_features_ = false;
    }

    RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been initialized");
}

VisualOdometryNode::~VisualOdometryNode() {
    if (show_original_) cv::destroyWindow(original_window_name_);
    if (show_features_) cv::destroyWindow(feature_window_name_);
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

    // 시각화 파라미터
    this->declare_parameter("visualization.window_width", 800);
    this->declare_parameter("visualization.window_height", 600);
    this->declare_parameter("visualization.show_original", true);
    this->declare_parameter("visualization.show_features", true);
    this->declare_parameter("visualization.window_pos_x", 100);
    this->declare_parameter("visualization.window_pos_y", 100);
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

    // 시각화 파라미터 적용
    window_width_ = this->get_parameter("visualization.window_width").as_int();
    window_height_ = this->get_parameter("visualization.window_height").as_int();
    show_original_ = this->get_parameter("visualization.show_original").as_bool();
    show_features_ = this->get_parameter("visualization.show_features").as_bool();
    window_pos_x_ = this->get_parameter("visualization.window_pos_x").as_int();
    window_pos_y_ = this->get_parameter("visualization.window_pos_y").as_int();

    // 시각화 타입 설정
    std::string viz_type = this->get_parameter("feature_detector.visualization_type").as_string();
    feature_detector_.setVisualizationType(viz_type);
    
    RCLCPP_INFO(this->get_logger(), 
                "Applied visualization type: %s", 
                viz_type.c_str());
}

rcl_interfaces::msg::SetParametersResult VisualOdometryNode::onParamChange(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& param : params) {
        if (param.get_name() == "feature_detector.max_features") {
            feature_detector_.setMaxFeatures(param.as_int());
            RCLCPP_INFO(this->get_logger(), "Updated max_features to: %ld", param.as_int());
        }
        else if (param.get_name() == "feature_detector.scale_factor") {
            feature_detector_.setScaleFactor(param.as_double());
        }
        else if (param.get_name() == "feature_detector.n_levels") {
            feature_detector_.setNLevels(param.as_int());
        }
        else if (param.get_name() == "feature_detector.visualization_type") {
            feature_detector_.setVisualizationType(param.as_string());
            RCLCPP_INFO(this->get_logger(), 
                       "Updated visualization type to: %s", 
                       param.as_string().c_str());
        }
        else if (param.get_name() == "visualization.window_width") {
            window_width_ = param.as_int();
        }
        else if (param.get_name() == "visualization.window_height") {
            window_height_ = param.as_int();
        }
        else if (param.get_name() == "visualization.show_original") {
            show_original_ = param.as_bool();
            if (show_original_) {
                cv::namedWindow(original_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
                cv::moveWindow(original_window_name_, window_pos_x_, window_pos_y_);
            } else {
                cv::destroyWindow(original_window_name_);
            }
        }
        else if (param.get_name() == "visualization.show_features") {
            show_features_ = param.as_bool();
            if (show_features_) {
                cv::namedWindow(feature_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
                cv::moveWindow(feature_window_name_, 
                             window_pos_x_ + window_width_ + 30,
                             window_pos_y_);
            } else {
                cv::destroyWindow(feature_window_name_);
            }
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
        auto frame_start_time = this->now();

        // 이미지 직접 변환 (jpeg 포맷 자동 처리)
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
        
        // 이미지 크기 확인 및 리사이즈 (한 번의 연산으로)
        static cv::Mat working_frame;  // 정적 버퍼
        if (cv_ptr->image.rows > 720) {
            if (working_frame.empty() || 
                working_frame.size() != cv::Size(cv_ptr->image.cols * 720.0 / cv_ptr->image.rows, 720)) {
                working_frame.create(720, cv_ptr->image.cols * 720.0 / cv_ptr->image.rows, cv_ptr->image.type());
            }
            cv::resize(cv_ptr->image, working_frame, working_frame.size(), 0, 0, cv::INTER_AREA);
        } else {
            working_frame = cv_ptr->image;
        }

        // 특징점 검출 (필요한 경우만)
        Features features;
        features_detected_ = false;  // 상태 업데이트
        if (show_features_ || feature_img_pub_->get_subscription_count() > 0) {
            static cv::Mat gray_buffer;  // 그레이스케일 변환용 정적 버퍼
            features = feature_detector_.detectFeatures(
                image_processor_.process(working_frame, gray_buffer).gray);
            features_detected_ = !features.keypoints.empty();
        }

        // 디스플레이 업데이트 (30Hz로 제한)
        static rclcpp::Time last_display_time = this->now();
        if ((this->now() - last_display_time).seconds() >= 1.0/30.0) {
            try {
                static cv::Mat display_buffer;  // 디스플레이용 정적 버퍼
                
                if (show_original_) {
                    if (working_frame.rows != window_height_ || 
                        working_frame.cols != window_width_) {
                        cv::resize(working_frame, display_buffer, 
                                 cv::Size(window_width_, window_height_), 
                                 0, 0, cv::INTER_NEAREST);
                        cv::imshow(original_window_name_, display_buffer);
                    } else {
                        cv::imshow(original_window_name_, working_frame);
                    }
                }
                
                if (show_features_ && !features.visualization.empty()) {
                    cv::resize(features.visualization, display_buffer, 
                             cv::Size(window_width_, window_height_), 
                             0, 0, cv::INTER_NEAREST);
                    cv::imshow(feature_window_name_, display_buffer);
                }
                
                cv::waitKey(1);
                last_display_time = this->now();
            } catch (const cv::Exception& e) {
                RCLCPP_WARN_THROTTLE(this->get_logger(),
                                   *this->get_clock(),
                                   5000,
                                   "Display error: %s", e.what());
            }
        }

        // 현재 프레임 저장 (필요한 경우만, 메모리 재사용)
        if (features_detected_) {
            if (previous_frame_.empty() || 
                previous_frame_.size() != working_frame.size()) {
                previous_frame_.create(working_frame.size(), working_frame.type());
                current_frame_.create(working_frame.size(), working_frame.type());
            }
            current_frame_.copyTo(previous_frame_);
            working_frame.copyTo(current_frame_);
        }

        // 실제 처리 결과 퍼블리시
        if (feature_img_pub_->get_subscription_count() > 0 && !features.visualization.empty()) {
            sensor_msgs::msg::Image::SharedPtr feature_msg = 
                cv_bridge::CvImage(msg->header, "bgr8", features.visualization)
                .toImageMsg();
            feature_img_pub_->publish(*feature_msg);
        }

        // 프레임 처리 시간 계산
        auto frame_end_time = this->now();
        double process_time = (frame_end_time - frame_start_time).seconds() * 1000.0;  // ms
        
        // FPS 계산을 위한 누적
        fps_frame_count_++;
        fps_total_process_time_ += process_time;
        
        // 1초마다 성능 통계 출력
        if (last_fps_time_.get_clock_type() == RCL_ROS_TIME) {  // 초기화 확인
            double elapsed_time = (frame_end_time - last_fps_time_).seconds();
            if (elapsed_time >= 1.0) {
                double fps = fps_frame_count_ / elapsed_time;
                double avg_process_time = fps_total_process_time_ / fps_frame_count_;
                
                RCLCPP_INFO(this->get_logger(),
                           "Performance Stats:\n"
                           "  FPS: %.1f\n"
                           "  Process Time: %.1f ms (avg)\n"
                           "  Features: %zu\n"
                           "  Frame Size: %dx%d\n"
                           "  Total Frames: %d\n"
                           "  Elapsed Time: %.3f s",
                           fps,
                           avg_process_time,
                           features.keypoints.size(),
                           working_frame.cols,
                           working_frame.rows,
                           fps_frame_count_,
                           elapsed_time);

                // 통계 리셋
                fps_frame_count_ = 0;
                fps_total_process_time_ = 0.0;
                last_fps_time_ = frame_end_time;
            }
        } else {
            last_fps_time_ = frame_end_time;  // 첫 프레임에서 초기화
        }

        // 개별 프레임 처리 시간 로깅 (5초마다)
        RCLCPP_DEBUG_THROTTLE(this->get_logger(),
                            *this->get_clock(),
                            5000,
                            "Current frame timing:\n"
                            "  Process time: %.1f ms\n"
                            "  Image timestamp: %.3f",
                            process_time,
                            msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
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
    if (!camera_info_received_ || !features_detected_) {
        return;  // 특징점이 검출된 경우에만 처리
    }

    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
        
        if (current_depth_.empty() || 
            current_depth_.size() != cv_ptr->image.size()) {
            current_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());
            previous_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());
        }
        
        current_depth_.copyTo(previous_depth_);
        cv_ptr->image.copyTo(current_depth_);
        
        RCLCPP_INFO(this->get_logger(), "Depth image processed. Size: %dx%d",
                    current_depth_.cols, current_depth_.rows);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error in depth callback: %s", e.what());
    }
}

} // namespace vo