#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "visual_odometry/msg/vo_state.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "visual_odometry/visualization.hpp"
#include "visual_odometry/frame_processor.hpp"
#include "visual_odometry/logger.hpp"
#include "visual_odometry/resource_monitor.hpp"

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
        
        // 로깅 객체 초기화 (다른 초기화보다 먼저)
        logger_ = std::make_unique<Logger>(this);
        resource_monitor_ = std::make_unique<ResourceMonitor>(this);
        
        // 초기 시스템 정보 로깅
        SystemInfo system_info;
        system_info.window_width = window_width_;
        system_info.window_height = window_height_;
        system_info.window_pos_x = window_pos_x_;
        system_info.window_pos_y = window_pos_y_;
        system_info.show_original = show_original_;
        system_info.show_features = show_features_;
        system_info.show_matches = show_matches_;
        
        // Feature Detector 설정
        system_info.max_features = this->get_parameter("feature_detector.max_features").as_int();
        system_info.scale_factor = this->get_parameter("feature_detector.scale_factor").as_double();
        system_info.n_levels = this->get_parameter("feature_detector.n_levels").as_int();
        
        // 입력 소스 설정 (yaml 반영 - applyCurrentParameters 전에 param 직접 읽기)
        system_info.input_source = this->get_parameter("input.source").as_string();
        system_info.rgb_topic = this->get_parameter("topics.rgb_image").as_string();
        system_info.depth_topic = this->get_parameter("topics.depth_image").as_string();
        system_info.camera_info_topic = this->get_parameter("topics.camera_info").as_string();
        
        // 처리 설정
        system_info.queue_size = 5;  // 기본 큐 크기
        system_info.target_fps = 60.0;  // 목표 FPS

        // 카메라 파라미터: camera_info 수신 전이므로 0 (로거에서 N/A 출력)
        system_info.camera_width = 0;
        system_info.camera_height = 0;
        system_info.camera_fx = 0.0;
        system_info.camera_fy = 0.0;
        system_info.camera_cx = 0.0;
        system_info.camera_cy = 0.0;
        
        // 시스템 정보 로깅
        logger_->logSystemInfo(system_info);

        // 3. Feature detector와 matcher를 shared_ptr로 초기화
        feature_detector_ = std::make_shared<FeatureDetector>();
        feature_matcher_ = std::make_shared<FeatureMatcher>();

        if (!feature_detector_ || !feature_matcher_) {
            throw std::runtime_error("Failed to initialize feature detector or matcher");
        }

        // 4. 나머지 파라미터 적용
        applyCurrentParameters();
    
        // QoS 프로파일 설정 (ZED sensor 토픽과 호환)
        auto qos = rclcpp::QoS(1).best_effort().durability_volatile();
        auto sensor_qos = rclcpp::SensorDataQoS();

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
            depth_topic, sensor_qos,
        std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, sensor_qos,
        std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));

        // Publishers
    feature_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            feature_topic, 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "camera_pose", 10);
        vo_state_pub_ = this->create_publisher<visual_odometry::msg::VOState>(
            "vo_state", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // 이전 프레임 관련 변수 초기화
        prev_frame_ = cv::Mat();
        prev_frame_gray_ = cv::Mat();
        prev_features_ = Features();
        first_frame_ = true;

        // Visualizer 초기화 (시각화 설정과 관계없이 항상 생성)
        visualizer_ = std::make_unique<Visualizer>();
        visualizer_->setWindowSize(window_width_, window_height_);
        visualizer_->setShowOriginal(show_original_);
        visualizer_->setShowFeatures(show_features_);
        visualizer_->setShowMatches(show_matches_);
        
        // 시각화가 활성화된 경우에만 윈도우 생성
        bool visualization_enabled = show_original_ || show_features_ || show_matches_;
        if (visualization_enabled) {
            visualizer_->createWindows();
        }

        // 프레임 처리 객체 초기화
        frame_processor_ = std::make_unique<FrameProcessor>(
            feature_detector_,
            feature_matcher_
        );

        // T_global 초기화 (4x4 identity)
        T_global_ = cv::Mat::eye(4, 4, CV_64F);

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
    
    // display_thread_ 관련 코드 제거
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    // 시각화 관련 정리
    if (visualizer_) {
        visualizer_->destroyWindows();
    }
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

    // VO 파라미터 (정지 시 드리프트 방지)
    this->declare_parameter("vo.zero_motion_threshold_mm", 2.0);
    this->declare_parameter("vo.zero_motion_rotation_threshold_rad", 0.002);
    this->declare_parameter("frames.frame_id", std::string("odom"));
    this->declare_parameter("frames.child_frame_id", std::string("camera_link"));
    this->declare_parameter("tf.publish", true);

    // 파라미터 값 로깅
    show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
    RCLCPP_INFO(this->get_logger(), "Initialized show_matches_: %s", show_matches_ ? "true" : "false");

    // 특징점 검출 파라미터 초기화
    max_features_ = this->get_parameter("feature_detector.max_features").as_int();
    fast_threshold_ = this->get_parameter("feature_detector.fast_threshold").as_int();
}

void VisualOdometryNode::applyCurrentParameters() {
    try {
        std::vector<rclcpp::Parameter> current_params = this->get_parameters(
            {"visualization.window_width", "visualization.window_height",
             "visualization.windows.show_original", "visualization.windows.show_features",
             "visualization.windows.show_matches", "visualization.window_pos_x",
             "visualization.window_pos_y", "feature_detector.max_features",
             "feature_detector.scale_factor", "feature_detector.n_levels",
             "feature_detector.visualization_type"});

        // 시각화 파라미터 적용
        window_width_ = this->get_parameter("visualization.window_width").as_int();
        window_height_ = this->get_parameter("visualization.window_height").as_int();
        show_original_ = this->get_parameter("visualization.windows.show_original").as_bool();
        show_features_ = this->get_parameter("visualization.windows.show_features").as_bool();
        show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
        window_pos_x_ = this->get_parameter("visualization.window_pos_x").as_int();
        window_pos_y_ = this->get_parameter("visualization.window_pos_y").as_int();

        // Logger를 통한 시각화 파라미터 로깅
        logger_->logVisualizationParameters(
            window_width_, window_height_,
            window_pos_x_, window_pos_y_,
            show_original_, show_features_,
            show_matches_
        );

        if (feature_detector_) {
            int max_features = this->get_parameter("feature_detector.max_features").as_int();
            double scale_factor = this->get_parameter("feature_detector.scale_factor").as_double();
            int n_levels = this->get_parameter("feature_detector.n_levels").as_int();
            std::string viz_type = this->get_parameter("feature_detector.visualization_type").as_string();

            // Logger를 통한 Feature Detector 파라미터 로깅
            logger_->logFeatureDetectorParameters(
                max_features, scale_factor, n_levels, viz_type
            );

            feature_detector_->setMaxFeatures(max_features);
            feature_detector_->setScaleFactor(scale_factor);
            feature_detector_->setNLevels(n_levels);
            feature_detector_->setVisualizationType(viz_type);
        }

        // 파라미터 변경 로깅도 Logger를 통해 처리
        for (const auto& param : current_params) {
            logger_->logParameterUpdate(param.get_name(), param);
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
            // zed_sdk 모드: 카메라 파라미터 설정 (camera_info 토픽 없음)
            cv::Mat K, D;
            int w, h;
            if (zed_interface_->getCameraParameters(K, D) && zed_interface_->getResolution(w, h)) {
                camera_params_.fx = K.at<double>(0, 0);
                camera_params_.fy = K.at<double>(1, 1);
                camera_params_.cx = K.at<double>(0, 2);
                camera_params_.cy = K.at<double>(1, 2);
                camera_params_.width = w;
                camera_params_.height = h;
                camera_info_received_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "Camera parameters (ZED SDK): %dx%d, fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                    w, h, camera_params_.fx, camera_params_.fy, camera_params_.cx, camera_params_.cy);
            }
        }

        // FPS 윈도우 크기 설정
        fps_window_size_ = this->get_parameter("fps_window_size").as_int();
        original_frame_times_.clear();  // 기존 데이터 초기화
        feature_frame_times_.clear();  // 기존 데이터 초기화
    }
    catch (const std::exception& e) {
        logger_->logError("ParameterUpdate", std::string("Error in applyCurrentParameters: ") + e.what());
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
            RCLCPP_INFO(this->get_logger(),
                "Camera parameters received: %dx%d, fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                camera_params_.width, camera_params_.height,
                camera_params_.fx, camera_params_.fy, camera_params_.cx, camera_params_.cy);
        }
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in camera info callback: %s", e.what());
    }
}

void VisualOdometryNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // zed_sdk 모드: 큐 경로 사용, rgb 콜백 무시
    if (input_source_ == "zed_sdk") {
        return;
    }

    try {
        // 큐 크기 업데이트
        {
            std::lock_guard<std::mutex> lock(image_queue_mutex_);
            resource_monitor_->updateQueueSize(image_queue_.size());
        }

        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
        cv::Mat rgb = cv_ptr->image;
        cv::Mat depth;

        // ros2 모드: 최신 depth와 페어링 (타임스탬프 근사 동기화)
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            if (!current_depth_.empty()) {
                depth = current_depth_.clone();
            }
        }

        resource_monitor_->checkResources();
        processImages(rgb, depth);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
    }
}

void VisualOdometryNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_source_ == "zed_sdk") {
        return;
    }

    try {
        auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);

        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (current_depth_.empty()) {
            current_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());
            prev_depth_.create(cv_ptr->image.size(), cv_ptr->image.type());
        }

        cv_ptr->image.copyTo(current_depth_);
        current_depth_.copyTo(prev_depth_);
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
        // depth 수신 확인 (한 번만 로그)
        static bool depth_logged = false;
        if (!depth_logged && !depth.empty()) {
            RCLCPP_INFO(this->get_logger(), "Depth received: %dx%d (RGB-D sync OK)",
                depth.cols, depth.rows);
            depth_logged = true;
        }

        auto start_time = std::chrono::steady_clock::now();

        auto result = frame_processor_->processFrame(rgb, depth, camera_params_, first_frame_);

        // 리소스 모니터링 업데이트
        resource_monitor_->updateProcessingTime(result.feature_detection_time);
        
        // 처리 결과 로깅
        ProcessingMetrics metrics;
        metrics.detection_time = result.feature_detection_time;
        metrics.matching_time = result.feature_matching_time;
        metrics.num_features = result.features.keypoints.size();
        metrics.num_matches = result.matches.matches.size();
        metrics.num_3d_points = static_cast<int>(result.matches.prev_points_3d.size());
        metrics.matching_ratio = result.matches.matches.size() / 
            static_cast<double>(result.features.keypoints.size() + 1e-6);
        metrics.pnp_success = result.pnp_success;
        metrics.pnp_inliers = result.pnp_inliers;

        // 포즈 누적: PnP (curr→prev) → T_prev_curr → T_global
        // zero_motion: |t| < thresh_mm AND |θ| < thresh_rad 이면 누적 스킵 (정지 시 드리프트 방지)
        if (result.pnp_success && !result.R.empty() && !result.t.empty()) {
            cv::Mat R_cp = result.R, t_cp = result.t;  // curr→prev
            double t_norm = cv::norm(t_cp);
            double trace_R = R_cp.at<double>(0,0) + R_cp.at<double>(1,1) + R_cp.at<double>(2,2);
            double rot_angle = std::acos(std::min(1.0, std::max(-1.0, (trace_R - 1.0) / 2.0)));
            double thresh_mm = this->get_parameter("vo.zero_motion_threshold_mm").as_double();
            double thresh_rad = this->get_parameter("vo.zero_motion_rotation_threshold_rad").as_double();
            if (t_norm >= thresh_mm || rot_angle >= thresh_rad) {
                cv::Mat R_pc = R_cp.t();
                cv::Mat t_pc = -R_pc * t_cp;  // prev→curr
                cv::Mat T_pc = cv::Mat::eye(4, 4, CV_64F);
                R_pc.copyTo(T_pc(cv::Rect(0, 0, 3, 3)));
                t_pc.copyTo(T_pc(cv::Rect(3, 0, 1, 3)));
                T_global_ = T_global_ * T_pc;
            }
        }
        // REP 103: camera optical (X right, Y down, Z fwd) → ROS body (X fwd, Y left, Z up)
        // 수정: X↔Y 스왑, Z 반전 → ros_x=-cam_x, ros_y=cam_z, ros_z=cam_y
        static const cv::Mat R_cam_to_ros = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 0, 1, 0, 1, 0);
        cv::Mat t_cam(3, 1, CV_64F);
        t_cam.at<double>(0, 0) = T_global_.at<double>(0, 3) / 1000.0;  // mm→m
        t_cam.at<double>(1, 0) = T_global_.at<double>(1, 3) / 1000.0;
        t_cam.at<double>(2, 0) = T_global_.at<double>(2, 3) / 1000.0;
        cv::Mat t_ros = R_cam_to_ros * t_cam;
        metrics.pose_x = -t_ros.at<double>(1, 0);  // X↔Y 스왑 + 부호 반전
        metrics.pose_y = -t_ros.at<double>(0, 0);
        metrics.pose_z = t_ros.at<double>(2, 0);
        cv::Mat R_cam = T_global_(cv::Rect(0, 0, 3, 3));
        cv::Mat R_ros = R_cam_to_ros * R_cam * R_cam_to_ros.t();
        // X↔Y 스왑: S*R*S^T, S=[[0,1,0],[1,0,0],[0,0,1]]
        static const cv::Mat S_xy = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, 0, 0, 0, 0, 1);
        R_ros = S_xy * R_ros * S_xy.t();
        double sy = std::sqrt(R_ros.at<double>(0,0)*R_ros.at<double>(0,0) + R_ros.at<double>(1,0)*R_ros.at<double>(1,0));
        const double eps = 1e-6;
        if (sy >= eps) {
            metrics.pose_roll = std::atan2(R_ros.at<double>(2,1), R_ros.at<double>(2,2));
            metrics.pose_pitch = std::atan2(-R_ros.at<double>(2,0), sy);
            metrics.pose_yaw = -std::atan2(R_ros.at<double>(1,0), R_ros.at<double>(0,0));  // yaw 부호 반전
        } else {
            metrics.pose_roll = std::atan2(-R_ros.at<double>(1,2), R_ros.at<double>(1,1));
            metrics.pose_pitch = std::atan2(-R_ros.at<double>(2,0), sy);
            metrics.pose_yaw = 0.0;
        }

        // 시각화 시간 측정
        auto viz_start = std::chrono::steady_clock::now();
        if (visualizer_) {
            visualizer_->visualize(rgb, result.features, result.matches, prev_frame_);
        }
        auto viz_end = std::chrono::steady_clock::now();
        metrics.visualization_time = std::chrono::duration<double, std::milli>(
            viz_end - viz_start).count();
        
        // 전체 처리 시간 계산
        auto end_time = std::chrono::steady_clock::now();
        metrics.processing_time = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        // 메모리 사용량과 큐 크기 추가
        metrics.memory_usage = resource_monitor_->getMemoryUsage();
        {
            std::lock_guard<std::mutex> lock(image_queue_mutex_);
            metrics.queue_size = image_queue_.size();
        }
        
        // 통합된 메트릭스 업데이트
        logger_->updateMetrics(metrics);

        // 결과 발행 (publish_results 파라미터)
        if (this->get_parameter("processing.publish_results").as_bool()) {
            publishResults(metrics);
        }
        
        first_frame_ = false;
        prev_frame_ = rgb.clone();
        
    } catch (const std::exception& e) {
        logger_->logError("ImageProcessor", e.what());
    }
}

void VisualOdometryNode::publishResults(const ProcessingMetrics& metrics) {
    auto stamp = this->now();
    std::string frame_id = this->get_parameter("frames.frame_id").as_string();

    // PoseStamped
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_id;
    pose_msg.pose.position.x = metrics.pose_x;
    pose_msg.pose.position.y = metrics.pose_y;
    pose_msg.pose.position.z = metrics.pose_z;
    // roll, pitch, yaw → quaternion (ZYX)
    double cr = std::cos(metrics.pose_roll * 0.5);
    double sr = std::sin(metrics.pose_roll * 0.5);
    double cp = std::cos(metrics.pose_pitch * 0.5);
    double sp = std::sin(metrics.pose_pitch * 0.5);
    double cy = std::cos(metrics.pose_yaw * 0.5);
    double sy = std::sin(metrics.pose_yaw * 0.5);
    pose_msg.pose.orientation.x = sr * cp * cy - cr * sp * sy;
    pose_msg.pose.orientation.y = cr * sp * cy + sr * cp * sy;
    pose_msg.pose.orientation.z = cr * cp * sy - sr * sp * cy;
    pose_msg.pose.orientation.w = cr * cp * cy + sr * sp * sy;
    pose_pub_->publish(pose_msg);

    // TF (RViz 시각화용)
    if (this->get_parameter("tf.publish").as_bool()) {
        std::string child_frame_id = this->get_parameter("frames.child_frame_id").as_string();
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = frame_id;
        tf_msg.child_frame_id = child_frame_id;
        tf_msg.transform.translation.x = metrics.pose_x;
        tf_msg.transform.translation.y = metrics.pose_y;
        tf_msg.transform.translation.z = metrics.pose_z;
        tf_msg.transform.rotation = pose_msg.pose.orientation;
        tf_broadcaster_->sendTransform(tf_msg);
    }

    // VOState
    visual_odometry::msg::VOState vo_msg;
    vo_msg.header.stamp = stamp;
    vo_msg.header.frame_id = frame_id;
    vo_msg.pose = pose_msg.pose;
    vo_msg.num_features = static_cast<uint32_t>(metrics.num_features);
    vo_msg.num_matches = static_cast<uint32_t>(metrics.num_matches);
    vo_msg.tracking_quality = static_cast<float>(metrics.pnp_success ?
        std::min(1.0, metrics.pnp_inliers / 100.0) : 0.0);
    vo_msg.scale_confidence = 1.0f;  // RGB-D: 고정 스케일
    vo_msg.processing_time = static_cast<float>(metrics.processing_time);
    vo_state_pub_->publish(vo_msg);
}

void VisualOdometryNode::processingLoop() {
    while (rclcpp::ok() && !should_exit_) {
        std::unique_lock<std::mutex> lock(image_queue_mutex_);
        image_ready_.wait(lock, [this]() { return !image_queue_.empty(); });
        
        // 큐 크기 모니터링
        size_t queue_size = image_queue_.size();
        resource_monitor_->updateQueueSize(queue_size);
        
        // 큐 상태 로깅
        if (queue_size > 5) {
            logger_->logWarning("Queue", 
                "Queue size is high: " + std::to_string(queue_size));
        }

        auto [rgb, depth] = image_queue_.front();
        image_queue_.pop();
        lock.unlock();

        processImages(rgb, depth);
        resource_monitor_->checkResources();
    }
}

} // namespace vo