#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include "visual_odometry/msg/vo_state.hpp"

namespace vo {

VisualOdometryNode::VisualOdometryNode()
    : Node("visual_odometry_node",
           rclcpp::NodeOptions().use_intra_process_comms(true)),
      original_fps_(0.0),
      feature_fps_(0.0),
      zed_acquisition_time_(0.0) {
    try {
        // 1. 파라미터 선언
        declareParameters();
        
        // 2. 시각화 관련 파라미터만 먼저 적용
        window_width_ = this->get_parameter("visualization.window_width").as_int();
        window_height_ = this->get_parameter("visualization.window_height").as_int();
        show_original_ = this->get_parameter("visualization.windows.show_original").as_bool();
        show_features_ = this->get_parameter("visualization.windows.show_features").as_bool();
        show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
        
        // 3. Feature detector와 matcher 초기화
        feature_detector_ = std::make_unique<FeatureDetector>();
        feature_matcher_ = std::make_unique<FeatureMatcher>();

        if (!feature_detector_ || !feature_matcher_) {
            throw std::runtime_error("Failed to initialize feature detector or matcher");
        }

        // 4. 나머지 파라미터 적용
        applyCurrentParameters();

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

        // Publishers
        feature_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            feature_topic, 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "camera_pose", 10);
        vo_state_pub_ = this->create_publisher<visual_odometry::msg::VOState>(
            "vo_state", 10);

        // 이전 프레임 관련 변수 초기화
        prev_frame_ = cv::Mat();
        prev_frame_gray_ = cv::Mat();
        prev_features_ = Features();
        first_frame_ = true;

        // 윈도우 생성
        if (show_original_ || show_features_ || show_matches_) {
            if (show_original_) {
                cv::namedWindow(original_window_name_, cv::WINDOW_AUTOSIZE);
            }
            if (show_features_) {
                cv::namedWindow(feature_window_name_, cv::WINDOW_AUTOSIZE);
            }
            if (show_matches_) {
                cv::namedWindow(matches_window_name_, cv::WINDOW_AUTOSIZE);
            }
            display_thread_ = std::thread(&VisualOdometryNode::displayLoop, this);
        }

        // ZED SDK 모드를 위한 타이머 추가
        if (input_source_ == "zed_sdk") {
            zed_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(16),  // 60Hz
                std::bind(&VisualOdometryNode::zedTimerCallback, this));
        }

        // 이미지 처리 스레드 시작
        processing_thread_ = std::thread(&VisualOdometryNode::processingLoop, this);

        RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been initialized");
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in constructor: %s", e.what());
        throw;
    }
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
    try {
        // 시각화 파라미터 먼저 적용
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

        // Feature Detector 파라미터는 feature_detector_가 초기화된 후에만 적용
        if (feature_detector_) {
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
            
            feature_detector_->setMaxFeatures(max_features);
            feature_detector_->setScaleFactor(scale_factor);
            feature_detector_->setNLevels(n_levels);

            // 시각화 타입 설정
            std::string viz_type = this->get_parameter("feature_detector.visualization_type").as_string();
            feature_detector_->setVisualizationType(viz_type);
            
            RCLCPP_INFO(this->get_logger(), 
                        "Visualization Type: %s", 
                        viz_type.c_str());
        }

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
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in applyCurrentParameters: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult VisualOdometryNode::onParamChange(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& param : params) {
        if (param.get_name() == "feature_detector.max_features") {
            feature_detector_->setMaxFeatures(param.as_int());
            RCLCPP_INFO(this->get_logger(), "Updated max_features to: %ld", param.as_int());
        }
        else if (param.get_name() == "feature_detector.scale_factor") {
            feature_detector_->setScaleFactor(param.as_double());
        }
        else if (param.get_name() == "feature_detector.n_levels") {
            feature_detector_->setNLevels(param.as_int());
        }
        else if (param.get_name() == "feature_detector.visualization_type") {
            feature_detector_->setVisualizationType(param.as_string());
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

void VisualOdometryNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        
        if (current_depth_.empty()) {
            current_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());
            prev_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());  // previous_depth_ -> prev_depth_
        }
        
        cv_ptr->image.copyTo(current_depth_);
        current_depth_.copyTo(prev_depth_);  // previous_depth_ -> prev_depth_
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
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
        if (rgb.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty RGB image received");
            return;
        }

        // RGB to Gray 변환
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        // 특징점 검출
        Features features;
        double feature_detection_time = 0.0;
        {
            auto detect_start = std::chrono::steady_clock::now();
            features = feature_detector_->detectFeatures(gray);
            feature_detection_time = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - detect_start).count();
        }

        // 매칭 수행
        FeatureMatches matches;
        double feature_matching_time = 0.0;
        if (!prev_frame_gray_.empty() && !first_frame_) {
            auto match_start = std::chrono::steady_clock::now();
            matches = feature_matcher_->match(prev_features_, features,
                                           prev_frame_gray_, gray);
            feature_matching_time = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - match_start).count();
        }

        // 시각화
        double visualization_time = 0.0;
        if (show_original_ || show_features_ || show_matches_) {
            auto viz_start = std::chrono::steady_clock::now();
            
            try {
                std::lock_guard<std::mutex> lock(frame_mutex_);

                if (show_original_) {
                    cv::Mat resized;
                    cv::resize(rgb, resized, cv::Size(window_width_, window_height_));
                    display_frame_original_ = resized.clone();
                }

                if (show_features_) {
                    cv::Mat feature_img = rgb.clone();
                    cv::drawKeypoints(feature_img, features.keypoints, feature_img,
                                    cv::Scalar(0, 255, 0),
                                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    cv::resize(feature_img, feature_img, 
                             cv::Size(window_width_, window_height_));
                    display_frame_features_ = feature_img.clone();
                }

                if (show_matches_ && !prev_frame_gray_.empty() && !first_frame_) {
                    try {
                        // 매칭 결과 유효성 검사
                        bool valid_matches = true;
                        for (const auto& match : matches.matches) {
                            if (match.queryIdx >= prev_features_.keypoints.size() ||
                                match.trainIdx >= features.keypoints.size()) {
                                valid_matches = false;
                                RCLCPP_WARN(this->get_logger(), 
                                    "Invalid match index: query=%d/%zu, train=%d/%zu",
                                    match.queryIdx, prev_features_.keypoints.size(),
                                    match.trainIdx, features.keypoints.size());
                                break;
                            }
                        }

                        if (valid_matches && !matches.matches.empty()) {
                            cv::Mat matches_img;
                            std::vector<cv::DMatch> good_matches;
                            
                            // 유효한 매치만 필터링
                            for (const auto& match : matches.matches) {
                                if (match.queryIdx < prev_features_.keypoints.size() &&
                                    match.trainIdx < features.keypoints.size()) {
                                    good_matches.push_back(match);
                                }
                            }

                            // 매칭 수를 제한
                            size_t max_matches_to_draw = 100;  // 최대 100개만 표시
                            if (good_matches.size() > max_matches_to_draw) {
                                std::sort(good_matches.begin(), good_matches.end(),
                                         [](const cv::DMatch& a, const cv::DMatch& b) {
                                             return a.distance < b.distance;
                                         });
                                good_matches.resize(max_matches_to_draw);
                            }

                            if (!good_matches.empty()) {
                                // 이미지 크기 조정
                                cv::Mat prev_resized, curr_resized;
                                cv::resize(prev_frame_, prev_resized, 
                                         cv::Size(window_width_, window_height_));
                                cv::resize(rgb, curr_resized, 
                                         cv::Size(window_width_, window_height_));

                                // 매칭 그리기
                                cv::drawMatches(prev_resized, prev_features_.keypoints,
                                              curr_resized, features.keypoints,
                                              good_matches, matches_img,
                                              cv::Scalar(0, 255, 0),  // 매칭 선 색상 (녹색)
                                              cv::Scalar(255, 0, 0),  // 키포인트 색상 (파란색)
                                              std::vector<char>(),    // 마스크
                                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS |
                                              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                                // 매칭 선 두께 조정을 위한 후처리
                                cv::Mat overlay = matches_img.clone();
                                for (const auto& match : good_matches) {
                                    cv::Point2f pt1 = prev_features_.keypoints[match.queryIdx].pt;
                                    cv::Point2f pt2 = features.keypoints[match.trainIdx].pt;
                                    // 크기 조정된 좌표로 변환
                                    pt1.x *= static_cast<float>(window_width_) / prev_frame_.cols;
                                    pt1.y *= static_cast<float>(window_height_) / prev_frame_.rows;
                                    pt2.x *= static_cast<float>(window_width_) / rgb.cols;
                                    pt2.y *= static_cast<float>(window_height_) / rgb.rows;
                                    pt2.x += window_width_;  // 오른쪽 이미지의 좌표 조정
                                    cv::line(overlay, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                                }
                                
                                cv::addWeighted(overlay, 0.5, matches_img, 0.5, 0, matches_img);
                                
                                // 결과 저장
                                display_frame_matches_ = matches_img.clone();
                            }
                        }
                    }
                    catch (const cv::Exception& e) {
                        RCLCPP_ERROR(this->get_logger(), "OpenCV error in visualization: %s", e.what());
                    }
                }
            }
            catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV error in visualization: %s", e.what());
            }

            visualization_time = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - viz_start).count();
        }

        // 현재 프레임 저장
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            prev_frame_ = rgb.clone();
            prev_frame_gray_ = gray.clone();
            prev_features_ = features;
        }

        first_frame_ = false;

        // 성능 로깅
        static rclcpp::Time last_log_time = this->now();
        static int frame_count = 0;
        frame_count++;

        auto current_time = this->now();
        if ((current_time - last_log_time).seconds() >= 1.0) {
            double elapsed_time = (current_time - last_log_time).seconds();
            double fps = frame_count / elapsed_time;
            
            RCLCPP_INFO(this->get_logger(), 
                "\n[Processing Performance]"
                "\n- Feature Detection: %.1f ms (%.1f FPS)"
                "\n- Feature Matching:  %.1f ms"
                "\n- Visualization:     %.1f ms"
                "\n[Detection Results]"
                "\n- Features: %zu"
                "\n- Matches:  %zu",
                feature_detection_time, fps,
                feature_matching_time,
                visualization_time,
                features.keypoints.size(),
                matches.matches.size());

            frame_count = 0;
            last_log_time = current_time;
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