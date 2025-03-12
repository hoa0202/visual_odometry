#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include "visual_odometry/msg/vo_state.hpp"

namespace vo {

VisualOdometryNode::VisualOdometryNode() 
    : Node("visual_odometry_node",
           rclcpp::NodeOptions()
               .use_intra_process_comms(true)),  // 프로세스 내 통신만 사용
      original_fps_(0.0),
      feature_fps_(0.0),
      zed_acquisition_time_(0.0)
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

    // 토픽 이름 가져오기
    std::string rgb_topic = this->get_parameter("topics.rgb_image").as_string();
    std::string depth_topic = this->get_parameter("topics.depth_image").as_string();
    std::string camera_info_topic = this->get_parameter("topics.camera_info").as_string();
    std::string feature_topic = this->get_parameter("topics.feature_image").as_string();

    // Subscribers
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        rgb_topic, qos,
        std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        depth_topic, 10,
        std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, 10,
        std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));

    // Publisher
    feature_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "feature_image", 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "camera_pose", 10);
    vo_state_pub_ = this->create_publisher<visual_odometry::msg::VOState>(
        "vo_state", 10);

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
        if (show_matches_) {  // 매칭 윈도우 추가
            cv::namedWindow(matches_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
            cv::moveWindow(matches_window_name_,
                          window_pos_x_ + (window_width_ + 30) * 2,  // 특징점 윈도우 오른쪽에 배치
                          window_pos_y_);
        }

        // 디스플레이 스레드 시작
        if (show_original_ || show_features_ || show_matches_) {  // show_matches_ 조건 추가
            display_thread_ = std::thread(&VisualOdometryNode::displayLoop, this);
        }
    } catch (const cv::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "Failed to create windows: %s", e.what());
        show_original_ = false;
        show_features_ = false;
        show_matches_ = false;  // 매칭 윈도우도 비활성화
    }

    // ZED SDK 모드를 위한 타이머 추가
    if (input_source_ == "zed_sdk") {
        zed_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(16),  // 60Hz를 위해 16ms로 수정
            std::bind(&VisualOdometryNode::zedTimerCallback, this));
    }

    // FPS 측정을 위한 변수들 초기화
    original_frame_times_.clear();
    feature_frame_times_.clear();
    last_fps_print_time_ = this->now();

    // 이미지 처리 스레드 시작
    processing_thread_ = std::thread(&VisualOdometryNode::processingLoop, this);

    RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been initialized");

    // 생성자에 다음 로그 추가
    RCLCPP_INFO(this->get_logger(), 
        "Loaded parameters:"
        "\n  show_matches: %s"
        "\n  show_original: %s"
        "\n  show_features: %s",
        show_matches_ ? "true" : "false",
        show_original_ ? "true" : "false",
        show_features_ ? "true" : "false");

    // 이전 프레임 관련 변수 초기화
    prev_frame_ = cv::Mat();
    prev_features_ = Features();
}

VisualOdometryNode::~VisualOdometryNode() {
    should_exit_ = true;
    if (display_thread_.joinable()) {
        display_thread_.join();
    }
    if (show_original_) cv::destroyWindow(original_window_name_);
    if (show_features_) cv::destroyWindow(feature_window_name_);
    if (show_matches_) cv::destroyWindow(matches_window_name_);
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
    this->declare_parameter("visualization.window_pos_x", 100);
    this->declare_parameter("visualization.window_pos_y", 100);
    
    // 시각화 윈도우 파라미터 (경로 수정)
    this->declare_parameter("visualization.windows.show_original", true);
    this->declare_parameter("visualization.windows.show_features", true);
    this->declare_parameter("visualization.windows.show_matches", true);  // true로 변경
    
    // 토픽 파라미터 선언
    this->declare_parameter("topics.rgb_image", "/zed/zed_node/rgb/image_rect_color");
    this->declare_parameter("topics.depth_image", "/zed/zed_node/depth/depth_registered");
    this->declare_parameter("topics.camera_info", "/zed/zed_node/rgb/camera_info");
    this->declare_parameter("topics.feature_image", "feature_image");

    // 입력 소스 파라미터
    this->declare_parameter("input.source", "ros2");
    this->declare_parameter("input.zed.serial_number", 0);
    this->declare_parameter("input.zed.resolution", "HD1080");
    this->declare_parameter("input.zed.fps", 30);
    this->declare_parameter("input.zed.depth_mode", "ULTRA");

    // FPS 윈도우 크기를 60으로 변경
    this->declare_parameter("fps_window_size", 60);  // 30에서 60으로 수정

    // 추가 Feature Detector 파라미터
    this->declare_parameter("feature_detector.fast_threshold", 25);
    this->declare_parameter("feature_detector.image_scale", 0.2);
    this->declare_parameter("feature_detector.matching.ratio_threshold", 0.8);
    this->declare_parameter("feature_detector.matching.cross_check", false);

    // 처리 관련 파라미터 추가
    this->declare_parameter("processing.enable_feature_detection", true);
    this->declare_parameter("processing.enable_feature_matching", true);
    this->declare_parameter("processing.enable_pose_estimation", true);
    this->declare_parameter("processing.publish_results", true);

    // 시각화 활성화 파라미터 추가
    this->declare_parameter("visualization.enable", true);

    // 파라미터 값 로깅
    show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
    RCLCPP_INFO(this->get_logger(), "Initialized show_matches_: %s", show_matches_ ? "true" : "false");

    // 특징점 검출 파라미터 초기화
    max_features_ = this->get_parameter("feature_detector.max_features").as_int();
    fast_threshold_ = this->get_parameter("feature_detector.fast_threshold").as_int();
}

void VisualOdometryNode::applyCurrentParameters() {
    // 현재 파라미터 값을 가져와서 적용
    int max_features = this->get_parameter("feature_detector.max_features").as_int();
    double scale_factor = this->get_parameter("feature_detector.scale_factor").as_double();
    int n_levels = this->get_parameter("feature_detector.n_levels").as_int();
    
    // Feature Detector 파라미터 로깅
    RCLCPP_INFO(this->get_logger(), 
                "Feature Detector Parameters:"
                "\n  - max_features: %d"
                "\n  - scale_factor: %.2f"
                "\n  - n_levels: %d",
                max_features, scale_factor, n_levels);
    
    feature_detector_.setMaxFeatures(max_features);
    feature_detector_.setScaleFactor(scale_factor);
    feature_detector_.setNLevels(n_levels);

    // 시각화 파라미터 적용 (경로 수정)
    window_width_ = this->get_parameter("visualization.window_width").as_int();
    window_height_ = this->get_parameter("visualization.window_height").as_int();
    show_original_ = this->get_parameter("visualization.windows.show_original").as_bool();
    show_features_ = this->get_parameter("visualization.windows.show_features").as_bool();
    show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
    window_pos_x_ = this->get_parameter("visualization.window_pos_x").as_int();
    window_pos_y_ = this->get_parameter("visualization.window_pos_y").as_int();

    // 시각화 파라미터 로깅
    RCLCPP_INFO(this->get_logger(), 
                "Visualization Parameters:"
                "\n  - window_size: %dx%d"
                "\n  - window_position: (%d, %d)"
                "\n  - show_original: %s"
                "\n  - show_features: %s"
                "\n  - show_matches: %s",
                window_width_, window_height_,
                window_pos_x_, window_pos_y_,
                show_original_ ? "true" : "false",
                show_features_ ? "true" : "false",
                show_matches_ ? "true" : "false");

    // 시각화 타입 설정
    std::string viz_type = this->get_parameter("feature_detector.visualization_type").as_string();
    feature_detector_.setVisualizationType(viz_type);
    
    RCLCPP_INFO(this->get_logger(), 
                "Visualization Type: %s", 
                viz_type.c_str());

    // 입력 소스 설정
    input_source_ = this->get_parameter("input.source").as_string();
    if (input_source_ == "zed_sdk") {
        zed_interface_ = std::make_unique<ZEDInterface>();
        int serial = this->get_parameter("input.zed.serial_number").as_int();
        std::string res = this->get_parameter("input.zed.resolution").as_string();
        int fps = this->get_parameter("input.zed.fps").as_int();
        std::string depth_mode = this->get_parameter("input.zed.depth_mode").as_string();
        
        sl::RESOLUTION resolution = sl::RESOLUTION::HD1080;  // 기본값
        if (res == "HD2K") resolution = sl::RESOLUTION::HD2K;
        else if (res == "HD720") resolution = sl::RESOLUTION::HD720;
        
        sl::DEPTH_MODE depth = sl::DEPTH_MODE::ULTRA;  // 기본값
        if (depth_mode == "PERFORMANCE") depth = sl::DEPTH_MODE::PERFORMANCE;
        else if (depth_mode == "QUALITY") depth = sl::DEPTH_MODE::QUALITY;
        
        if (!zed_interface_->connect(serial, resolution, fps, depth)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to ZED camera");
            return;
        }
    }

    // FPS 윈도우 크기 설정
    fps_window_size_ = this->get_parameter("fps_window_size").as_int();
    original_frame_times_.clear();  // 기존 데이터 초기화
    feature_frame_times_.clear();  // 기존 데이터 초기화
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
        else if (param.get_name() == "visualization.windows.show_original") {
            show_original_ = param.as_bool();
            if (show_original_) {
                cv::namedWindow(original_window_name_, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
                cv::moveWindow(original_window_name_, window_pos_x_, window_pos_y_);
            } else {
                cv::destroyWindow(original_window_name_);
            }
        }
        else if (param.get_name() == "visualization.windows.show_features") {
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

void VisualOdometryNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    try {
        if (!camera_info_received_) {
            // 카메라 파라미터 설정
            camera_params_.fx = msg->k[0];  // fx
            camera_params_.fy = msg->k[4];  // fy
            camera_params_.cx = msg->k[2];  // cx
            camera_params_.cy = msg->k[5];  // cy
            camera_params_.width = msg->width;
            camera_params_.height = msg->height;

            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera parameters received");
        }
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in camera info callback: %s", e.what());
    }
}

void VisualOdometryNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_source_ != "ros2") return;

    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
        cv::Mat rgb = cv_ptr->image;
        cv::Mat depth;  // depth는 depth 콜백에서 처리
        
        // 공통 이미지 처리 함수 호출
        processImages(rgb, depth);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
    }
}

void VisualOdometryNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (input_source_ != "ros2") return;  // ZED SDK 모드에서는 콜백 무시
    
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

bool VisualOdometryNode::getImages(cv::Mat& rgb, cv::Mat& depth) {
    if (input_source_ == "zed_sdk") {
        if (!zed_interface_ || !zed_interface_->isConnected()) {
            RCLCPP_ERROR(this->get_logger(), "ZED camera is not connected!");
            return false;
        }
        return zed_interface_->getImages(rgb, depth);
    }
    
    // ROS2 토픽을 통해 이미지를 받는 경우는 기존 콜백에서 처리
    return false;
}

void VisualOdometryNode::zedTimerCallback() {
    cv::Mat rgb, depth;
    if (getImages(rgb, depth)) {
        // 이미지를 큐에 추가만 하고 바로 리턴
        {
            std::lock_guard<std::mutex> lock(image_queue_mutex_);
            image_queue_.push({rgb.clone(), depth.clone()});
            if (image_queue_.size() > 2) {  // 최대 2개의 프레임만 유지
                image_queue_.pop();
            }
        }
        image_ready_.notify_one();
    }
}

void VisualOdometryNode::processImages(const cv::Mat& rgb, const cv::Mat& depth) {
    try {
        // 입력 이미지 체크
        if (rgb.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty RGB image received");
            return;
        }

        // RGB to Gray 변환
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        // 특징점 검출
        auto features = feature_detector_.detectFeatures(gray,
            max_features_, fast_threshold_);

        // 매칭 수행 (첫 프레임이 아닐 때만)
        if (!prev_frame_.empty()) {
            auto matches = feature_detector_.matchFeatures(prev_features_, features);

            // 시각화 업데이트
            if (show_original_ || show_features_ || show_matches_) {
                cv::Mat visualization;
                
                // 원본 이미지 표시
                if (show_original_) {
                    cv::Mat resized;
                    cv::resize(rgb, resized, cv::Size(window_width_, window_height_));
                    cv::imshow(original_window_name_, resized);
                    cv::moveWindow(original_window_name_, window_pos_x_, window_pos_y_);
                }

                // 특징점 표시
                if (show_features_) {
                    cv::Mat feature_img = rgb.clone();
                    for (const auto& kp : features.keypoints) {
                        cv::circle(feature_img, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
                    }
                    cv::resize(feature_img, feature_img, cv::Size(window_width_, window_height_));
                    cv::imshow(feature_window_name_, feature_img);
                    cv::moveWindow(feature_window_name_, window_pos_x_ + window_width_ + 10, window_pos_y_);
                }

                // 매칭 결과 표시
                if (show_matches_) {
                    cv::Mat matches_img;
                    cv::drawMatches(prev_frame_, prev_features_.keypoints,
                                  rgb, features.keypoints,
                                  matches.matches, matches_img,
                                  cv::Scalar::all(-1), cv::Scalar::all(-1),
                                  std::vector<char>(),
                                  cv::DrawMatchesFlags::DEFAULT);
                    cv::resize(matches_img, matches_img, cv::Size(window_width_, window_height_));
                    cv::imshow(matches_window_name_, matches_img);
                    cv::moveWindow(matches_window_name_, 
                                 window_pos_x_ + 2 * (window_width_ + 10), 
                                 window_pos_y_);
                }
                
                cv::waitKey(1);
            }

            // 결과 발행
            publishResults(features, matches);
        }

        // 현재 프레임 저장
        prev_frame_ = rgb.clone();
        prev_features_ = features;

        RCLCPP_INFO(this->get_logger(), 
            "Frame processed: %d features detected", 
            static_cast<int>(features.keypoints.size()));
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in processImages: %s", e.what());
    }
}

void VisualOdometryNode::displayLoop() {
    static bool was_showing_matches = false;  // 이전 상태 저장

    while (rclcpp::ok() && !should_exit_) {
        if (show_original_ || show_features_ || show_matches_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            
            // 한 번에 모든 윈도우 업데이트
            if (show_original_ && !display_frame_original_.empty()) {
                cv::imshow(original_window_name_, display_frame_original_);
            }
            if (show_features_ && !display_frame_features_.empty()) {
                cv::imshow(feature_window_name_, display_frame_features_);
            }
            if (show_matches_ && !display_frame_matches_.empty()) {
                cv::imshow(matches_window_name_, display_frame_matches_);
                was_showing_matches = true;
            } else if (was_showing_matches) {
                // 매칭 윈도우가 표시되었다가 비활성화된 경우 윈도우 닫기
                cv::destroyWindow(matches_window_name_);
                was_showing_matches = false;
            }
            
            cv::waitKey(1);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void VisualOdometryNode::processingLoop() {
    while (rclcpp::ok() && !should_exit_) {
        std::unique_lock<std::mutex> lock(image_queue_mutex_);
        image_ready_.wait(lock, [this]() { return !image_queue_.empty(); });

        auto [rgb, depth] = image_queue_.front();
        image_queue_.pop();
        lock.unlock();

        processImages(rgb, depth);
    }
}

void VisualOdometryNode::updateVisualization(const cv::Mat& frame,
                                           const Features& features,
                                           const FeatureMatches& matches) {
    try {
        std::lock_guard<std::mutex> lock(frame_mutex_);  // 스레드 안전성 보장

        // 버퍼 재사용을 위한 크기 확인 및 할당
        cv::Size display_size(window_width_, window_height_);
        
        if (display_frame_original_.empty() || display_frame_original_.size() != display_size) {
            display_frame_original_.create(display_size, CV_8UC3);
        }
        if (display_frame_features_.empty() || display_frame_features_.size() != display_size) {
            display_frame_features_.create(display_size, CV_8UC3);
        }
        if (display_frame_matches_.empty() || display_frame_matches_.size() != display_size) {
            display_frame_matches_.create(display_size, CV_8UC3);
        }

        // 원본 이미지 표시
        if (show_original_) {
            cv::resize(frame, display_frame_original_, display_size);
            cv::imshow(original_window_name_, display_frame_original_);
            cv::moveWindow(original_window_name_, window_pos_x_, window_pos_y_);
        }

        // 특징점 표시
        if (show_features_) {
            frame.copyTo(display_frame_features_);
            for (const auto& kp : features.keypoints) {
                cv::circle(display_frame_features_, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
            }
            cv::resize(display_frame_features_, display_frame_features_, display_size);
            cv::imshow(feature_window_name_, display_frame_features_);
            cv::moveWindow(feature_window_name_, 
                         window_pos_x_ + window_width_ + 10, 
                         window_pos_y_);
        }

        // 매칭 결과 표시
        if (show_matches_ && !matches.matches.empty()) {
            frame.copyTo(display_frame_matches_);
            for (size_t i = 0; i < matches.matches.size(); i++) {
                cv::line(display_frame_matches_,
                        matches.prev_points[i],
                        matches.curr_points[i],
                        cv::Scalar(0, 255, 0), 2);
                cv::circle(display_frame_matches_,
                          matches.curr_points[i], 3,
                          cv::Scalar(0, 0, 255), -1);
            }
            cv::resize(display_frame_matches_, display_frame_matches_, display_size);
            cv::imshow(matches_window_name_, display_frame_matches_);
            cv::moveWindow(matches_window_name_,
                         window_pos_x_ + 2 * (window_width_ + 10),
                         window_pos_y_);
        }

        cv::waitKey(1);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in updateVisualization: %s", e.what());
    }
}

void VisualOdometryNode::publishResults(const Features& features, 
                                      const FeatureMatches& matches) {
    try {
        auto current_time = this->now();
        
        // VO 상태 메시지 발행 (더 가벼운 메시지부터 처리)
        if (vo_state_pub_->get_subscription_count() > 0) {
            auto msg = visual_odometry::msg::VOState();
            msg.header.stamp = current_time;
            msg.header.frame_id = "camera_frame";
            
            // 기본 정보 설정
            msg.num_features = features.keypoints.size();
            msg.num_matches = matches.matches.size();
            
            // 품질 메트릭 계산 (0.0 ~ 1.0)
            msg.tracking_quality = features.keypoints.empty() ? 0.0f :
                static_cast<float>(matches.matches.size()) / 
                static_cast<float>(features.keypoints.size());
            
            // 처리 시간 설정
            msg.processing_time = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start_time_).count());
            
            vo_state_pub_->publish(msg);
        }

        // 특징점 이미지 발행 (구독자가 있을 때만)
        if (feature_img_pub_->get_subscription_count() > 0 && 
            !features.visualization.empty()) {
            
            sensor_msgs::msg::Image::SharedPtr feature_msg = 
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", 
                                 features.visualization).toImageMsg();
            feature_msg->header.stamp = current_time;
            feature_msg->header.frame_id = "camera_frame";
            
            feature_img_pub_->publish(*feature_msg);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in publishResults: %s", e.what());
    }
}

} // namespace vo