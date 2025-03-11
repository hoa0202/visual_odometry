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
        start_time_ = std::chrono::steady_clock::now();
        
        // 연산 활성화 상태 확인
        bool enable_detection = this->get_parameter("processing.enable_feature_detection").as_bool();
        bool enable_matching = this->get_parameter("processing.enable_feature_matching").as_bool();
        bool enable_visualization = this->get_parameter("visualization.enable").as_bool();
        
        Features curr_features;
        FeatureMatches matches;
        
        // 특징점 검출 (활성화된 경우만)
        if (enable_detection) {
            double image_scale = this->get_parameter("feature_detector.image_scale").as_double();
            int max_features = this->get_parameter("feature_detector.max_features").as_int();
            int fast_threshold = this->get_parameter("feature_detector.fast_threshold").as_int();
            
            // 작업용 프레임 준비
            cv::resize(rgb, working_frame_, cv::Size(), image_scale, image_scale);
            
            static cv::Mat gray;
            cv::cvtColor(working_frame_, gray, cv::COLOR_BGR2GRAY);
            
            curr_features = feature_detector_.detectFeatures(gray, max_features, fast_threshold);
            
            // 특징점 매칭 (활성화된 경우만)
            if (enable_matching && !prev_features_.keypoints.empty()) {
                float ratio_threshold = this->get_parameter(
                    "feature_detector.matching.ratio_threshold").as_double();
                bool cross_check = this->get_parameter(
                    "feature_detector.matching.cross_check").as_bool();
                
                matches = feature_detector_.matchFeatures(
                    prev_features_, curr_features, ratio_threshold, cross_check);
                
                // 매칭 로그를 DEBUG 레벨로 변경
                RCLCPP_DEBUG(this->get_logger(),
                    "Matching: %zu features matched between frames",
                    matches.matches.size());
            }
            
            // 현재 프레임을 이전 프레임으로 저장
            prev_features_ = curr_features;
            working_frame_.copyTo(prev_frame_);
            first_frame_ = false;
        }

        // 시각화 (활성화된 경우만)
        if (enable_visualization) {
            bool show_original = this->get_parameter("visualization.windows.show_original").as_bool();
            bool show_features = this->get_parameter("visualization.windows.show_features").as_bool();
            bool show_matches = this->get_parameter("visualization.windows.show_matches").as_bool();
            
            updateVisualization(rgb, curr_features, matches);
        }

        // 결과 발행 (활성화된 경우만)
        if (this->get_parameter("processing.publish_results").as_bool()) {
            publishResults(curr_features, matches);
        }

        // 시간 계산 (milliseconds로 변환)
        auto to_ms = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };

        double total_duration = to_ms(std::chrono::steady_clock::now() - start_time_);

        // 현재 시간 가져오기
        auto current_time = this->now();

        // 원본 이미지 FPS 계산 및 표시
        if (show_original_) {
            original_frame_times_.push_back(current_time.seconds());
            if (original_frame_times_.size() > static_cast<size_t>(fps_window_size_)) {
                original_frame_times_.pop_front();
            }
            if (original_frame_times_.size() >= 2) {
                double time_diff = original_frame_times_.back() - original_frame_times_.front();
                original_fps_ = (original_frame_times_.size() - 1) / time_diff;

                // 화면에 FPS 표시
                std::string fps_text = cv::format("FPS: %.1f", original_fps_);
                cv::putText(display_frame_original_, fps_text, 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                           1.0, cv::Scalar(0, 255, 0), 2);
            }
        }

        // 특징점 검출 FPS 계산 및 표시
        if (show_features_) {
            feature_frame_times_.push_back(current_time.seconds());
            if (feature_frame_times_.size() > static_cast<size_t>(fps_window_size_)) {
                feature_frame_times_.pop_front();
            }
            if (feature_frame_times_.size() >= 2) {
                double time_diff = feature_frame_times_.back() - feature_frame_times_.front();
                feature_fps_ = (feature_frame_times_.size() - 1) / time_diff;

                // 화면에 FPS 표시
                std::string fps_text = cv::format("FPS: %.1f", feature_fps_);
                cv::putText(display_frame_features_, fps_text, 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                           1.0, cv::Scalar(0, 255, 0), 2);
            }
        }

        // FPS 및 타이밍 정보 출력 (1초마다)
        if ((current_time - last_fps_print_time_).seconds() >= 1.0) {
            RCLCPP_INFO(this->get_logger(), 
                       "FPS - Original: %.1f, Features: %.1f, Matches: %.1f (Processing: %.1f ms, ZED: %.1f ms)",
                       original_fps_,
                       feature_fps_,
                       static_cast<double>(matches.matches.size()) / total_duration * 1000.0,  // 매칭 FPS 계산
                       total_duration,
                       zed_acquisition_time_);
            last_fps_print_time_ = current_time;

            // 타이밍 정보도 출력
            RCLCPP_INFO(this->get_logger(),
                        "Timing breakdown:\n"
                        "  Total: %.1f ms\n"
                        "  Matches: %zu",
                        total_duration,
                        matches.matches.size());
        }

        // 특징점 시각화 이미지 발행
        if (feature_img_pub_->get_subscription_count() > 0 && !curr_features.keypoints.empty()) {
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "zed_camera";
            
            sensor_msgs::msg::Image::SharedPtr feature_msg = 
                cv_bridge::CvImage(header, "bgr8", 
                                 curr_features.visualization).toImageMsg();
            feature_img_pub_->publish(*feature_msg);
        }
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

void VisualOdometryNode::updateVisualization(const cv::Mat& rgb, 
                                           const Features& features,
                                           const FeatureMatches& matches) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // 원본 이미지 시각화
    if (show_original_) {
        cv::resize(rgb, display_frame_original_,
                  cv::Size(window_width_, window_height_));
    }
    
    // 특징점 시각화
    if (show_features_ && !features.visualization.empty()) {
        cv::resize(features.visualization, display_frame_features_,
                  cv::Size(window_width_, window_height_));
    }
    
    // 매칭 결과 시각화
    if (show_matches_ && !matches.matches.empty() && !prev_frame_.empty()) {
        try {
            // 매칭 결과 시각화를 위한 임시 이미지
            cv::Mat match_vis;
            
            // 유효한 매칭만 필터링
            std::vector<cv::DMatch> good_matches;
            for (const auto& match : matches.matches) {
                if (match.queryIdx < prev_features_.keypoints.size() &&
                    match.trainIdx < features.keypoints.size()) {
                    good_matches.push_back(match);
                }
            }
            
            if (!good_matches.empty()) {
                cv::drawMatches(prev_frame_, prev_features_.keypoints,
                              working_frame_, features.keypoints,
                              good_matches, match_vis,
                              cv::Scalar::all(-1), cv::Scalar::all(-1),
                              std::vector<char>(),
                              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                
                cv::resize(match_vis, display_frame_matches_,
                          cv::Size(window_width_, window_height_));
            }
        }
        catch (const cv::Exception& e) {
            RCLCPP_WARN(this->get_logger(), 
                "Failed to visualize matches: %s", e.what());
        }
    }
}

void VisualOdometryNode::publishResults(const Features& features, 
                                      const FeatureMatches& matches) {
    // 특징점 이미지 발행
    if (feature_img_pub_->get_subscription_count() > 0 && !features.visualization.empty()) {
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_frame";
        
        sensor_msgs::msg::Image::SharedPtr feature_msg = 
            cv_bridge::CvImage(header, "bgr8", features.visualization).toImageMsg();
        feature_img_pub_->publish(*feature_msg);
    }
    
    // VO 상태 메시지 발행
    if (vo_state_pub_->get_subscription_count() > 0) {
        auto msg = visual_odometry::msg::VOState();
        msg.header.stamp = this->now();
        msg.header.frame_id = "camera_frame";
        
        // 특징점 정보 설정
        msg.num_features = features.keypoints.size();
        msg.num_matches = matches.matches.size();
        
        // 품질 메트릭 계산
        float tracking_quality = 0.0f;
        if (!features.keypoints.empty()) {
            tracking_quality = static_cast<float>(matches.matches.size()) / 
                             static_cast<float>(features.keypoints.size());
        }
        msg.tracking_quality = tracking_quality;
        
        // TODO: 스케일 신뢰도와 포즈 추정은 나중에 구현
        msg.scale_confidence = 0.0f;
        
        // 처리 시간 설정 - processImages에서 계산된 값 사용
        auto current_time = std::chrono::steady_clock::now();
        double processing_time = std::chrono::duration<double, std::milli>(
            current_time - start_time_).count();
        msg.processing_time = static_cast<float>(processing_time);
        
        vo_state_pub_->publish(msg);
    }
}

} // namespace vo