#include "visual_odometry/vo_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sl/Camera.hpp>

namespace vo {

VisualOdometryNode::VisualOdometryNode()
    : Node("visual_odometry_node",
           rclcpp::NodeOptions().use_intra_process_comms(true))
{
    try {
        declareParameters();

        window_width_ = this->get_parameter("visualization.window_width").as_int();
        window_height_ = this->get_parameter("visualization.window_height").as_int();
        show_original_ = this->get_parameter("visualization.windows.show_original").as_bool();
        show_features_ = this->get_parameter("visualization.windows.show_features").as_bool();
        show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();

        logger_ = std::make_unique<Logger>(this);
        resource_monitor_ = std::make_unique<ResourceMonitor>(this);

        SystemInfo system_info;
        system_info.window_width = window_width_;
        system_info.window_height = window_height_;
        system_info.window_pos_x = window_pos_x_;
        system_info.window_pos_y = window_pos_y_;
        system_info.show_original = show_original_;
        system_info.show_features = show_features_;
        system_info.show_matches = show_matches_;
        system_info.max_features = this->get_parameter("feature_detector.max_features").as_int();
        system_info.scale_factor = this->get_parameter("feature_detector.scale_factor").as_double();
        system_info.n_levels = this->get_parameter("feature_detector.n_levels").as_int();
        system_info.input_source = this->get_parameter("input.source").as_string();
        system_info.rgb_topic = this->get_parameter("topics.rgb_image").as_string();
        system_info.depth_topic = this->get_parameter("topics.depth_image").as_string();
        system_info.camera_info_topic = this->get_parameter("topics.camera_info").as_string();
        system_info.queue_size = 5;
        system_info.target_fps = 60.0;
        system_info.camera_width = 0;
        system_info.camera_height = 0;
        system_info.camera_fx = 0.0;
        system_info.camera_fy = 0.0;
        system_info.camera_cx = 0.0;
        system_info.camera_cy = 0.0;
        logger_->logSystemInfo(system_info);

        applyCurrentParameters();

        // ─── ROS2 I/O ──────────────────────────────────────────────────────
        auto qos = rclcpp::QoS(1).best_effort().durability_volatile();
        auto sensor_qos = rclcpp::SensorDataQoS();

        std::string rgb_topic = this->get_parameter("topics.rgb_image").as_string();
        std::string depth_topic = this->get_parameter("topics.depth_image").as_string();
        std::string camera_info_topic = this->get_parameter("topics.camera_info").as_string();
        std::string feature_topic = this->get_parameter("topics.feature_image").as_string();

        if (input_source_ == "ros2") {
            rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                rgb_topic, qos,
                std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));
            depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                depth_topic, sensor_qos,
                std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));
            camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic, sensor_qos,
                std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));
            std::string imu_sub_topic = this->get_parameter("topics.imu_sub").as_string();
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_sub_topic, rclcpp::SensorDataQoS(),
                std::bind(&VisualOdometryNode::imuCallback, this, std::placeholders::_1));
        }

        feature_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(feature_topic, 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("camera_pose", 10);
        vo_state_pub_ = this->create_publisher<visual_odometry::msg::VOState>("vo_state", 10);
        if (input_source_ == "zed_sdk") {
            imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
                this->get_parameter("topics.imu").as_string(), 10);
        }
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
        publishStaticTransform();

        // ─── Visualizer ─────────────────────────────────────────────────────
        visualizer_ = std::make_unique<Visualizer>();
        visualizer_->setWindowSize(window_width_, window_height_);
        visualizer_->setShowOriginal(show_original_);
        visualizer_->setShowFeatures(show_features_);
        visualizer_->setShowMatches(show_matches_);
        bool viz_enabled = show_original_ || show_features_ || show_matches_;
        if (viz_enabled) visualizer_->createWindows();

        // ─── VIOManager will be created once camera params are available ────
        // (zed_sdk: after applyCurrentParameters, ros2: after cameraInfoCallback)
        if (camera_info_received_) {
            vio_manager_ = std::make_unique<VIOManager>(
                loadVIOConfig(), this->get_logger(), this->get_clock());
            vio_manager_->setCameraParams(camera_params_);
        }

        // ─── ZED SDK ────────────────────────────────────────────────────────
        if (input_source_ == "zed_sdk") {
            zed_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(16),
                std::bind(&VisualOdometryNode::zedTimerCallback, this));
            imu_poll_thread_ = std::thread(&VisualOdometryNode::imuPollLoop, this);
        }

        processing_thread_ = std::thread(&VisualOdometryNode::processingLoop, this);
        fps_window_size_ = this->get_parameter("fps_window_size").as_int();

        RCLCPP_INFO(this->get_logger(), "Visual Odometry Node initialized (RTAB-Map VIO)");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Constructor error: %s", e.what());
        throw;
    }
}

VisualOdometryNode::~VisualOdometryNode() {
    should_exit_ = true;
    image_ready_.notify_all();
    if (imu_poll_thread_.joinable()) imu_poll_thread_.join();
    if (processing_thread_.joinable()) processing_thread_.join();
    if (visualizer_) visualizer_->destroyWindows();
}

// ─── Parameters ─────────────────────────────────────────────────────────────

void VisualOdometryNode::declareParameters() {
    // Feature detector
    this->declare_parameter("feature_detector.max_features", 2000);
    this->declare_parameter("feature_detector.scale_factor", 1.2);
    this->declare_parameter("feature_detector.n_levels", 8);
    this->declare_parameter("feature_detector.visualization_type", "points");
    this->declare_parameter("feature_detector.fast_threshold", 25);
    this->declare_parameter("feature_detector.image_scale", 0.2);
    this->declare_parameter("feature_detector.matching.ratio_threshold", 0.8);
    this->declare_parameter("feature_detector.matching.cross_check", false);

    // Image processor
    this->declare_parameter("image_processor.gaussian_blur_size", 5);
    this->declare_parameter("image_processor.gaussian_sigma", 1.0);
    this->declare_parameter("image_processor.enable_histogram_eq", true);

    // Visualization
    this->declare_parameter("visualization.window_width", 800);
    this->declare_parameter("visualization.window_height", 600);
    this->declare_parameter("visualization.window_pos_x", 100);
    this->declare_parameter("visualization.window_pos_y", 100);
    this->declare_parameter("visualization.windows.show_original", true);
    this->declare_parameter("visualization.windows.show_features", true);
    this->declare_parameter("visualization.windows.show_matches", true);
    this->declare_parameter("visualization.enable", true);

    // Topics
    this->declare_parameter("topics.rgb_image", "/zed/zed_node/rgb/image_rect_color");
    this->declare_parameter("topics.depth_image", "/zed/zed_node/depth/depth_registered");
    this->declare_parameter("topics.camera_info", "/zed/zed_node/rgb/camera_info");
    this->declare_parameter("topics.feature_image", "feature_image");
    this->declare_parameter("topics.imu", "imu");
    this->declare_parameter("topics.imu_sub", "/zed/zed_node/imu/data");

    // Input
    this->declare_parameter("input.source", "ros2");
    this->declare_parameter("input.zed.serial_number", 0);
    this->declare_parameter("input.zed.resolution", "HD1080");
    this->declare_parameter("input.zed.fps", 30);
    this->declare_parameter("input.zed.depth_mode", "ULTRA");

    // Processing
    this->declare_parameter("processing.enable_feature_detection", true);
    this->declare_parameter("processing.enable_feature_matching", true);
    this->declare_parameter("processing.enable_pose_estimation", true);
    this->declare_parameter("processing.publish_results", true);

    // Frames / TF
    this->declare_parameter("frames.frame_id", std::string("odom"));
    this->declare_parameter("frames.child_frame_id", std::string("camera_link"));
    this->declare_parameter("tf.publish", true);

    // FPS
    this->declare_parameter("fps_window_size", 60);

    // Legacy IMU params (kept for backward compat, VIO params below take priority)
    this->declare_parameter("imu.fusion_mode", std::string("none"));
    this->declare_parameter("imu.complementary_alpha", 0.98);
    this->declare_parameter("imu.enable", true);
    this->declare_parameter("imu.factor_graph_window_size", 20);
    this->declare_parameter("imu.ekf.chi2_threshold", 16.8);
    this->declare_parameter("imu.ekf.huber_pos_m", 0.1);
    this->declare_parameter("imu.ekf.huber_rot_rad", 0.1);

    // VO params (legacy)
    this->declare_parameter("vo.zero_motion_threshold_mm", 2.0);
    this->declare_parameter("vo.zero_motion_rotation_threshold_rad", 0.002);

    // ─── VIO Parameters (RTAB-Map style) ────────────────────────────────────
    this->declare_parameter("vio.init_mode", std::string("full"));
    this->declare_parameter("vio.use_imu", true);
    this->declare_parameter("vio.graph_window_size", 50);
    this->declare_parameter("vio.lost_timeout_sec", 5.0);
    this->declare_parameter("vio.vocabulary_path", std::string(""));
    this->declare_parameter("vio.loop_closure.enable", true);
    this->declare_parameter("vio.loop_closure.min_matches", 30);
    this->declare_parameter("vio.loop_closure.min_score", 0.3);
    this->declare_parameter("vio.loop_closure.min_interval_keyframes", 10);
    this->declare_parameter("vio.loop_closure.bayesian_threshold", 0.55);
    this->declare_parameter("vio.loop_closure.temporal_consistency", 3);
    this->declare_parameter("vio.memory.stm_size", 10);
    this->declare_parameter("vio.memory.rehearsal_similarity", 0.8);
    this->declare_parameter("vio.memory.proximity_max_dist", 1.0);
    this->declare_parameter("vio.imu_noise.accel_sigma", 0.05);
    this->declare_parameter("vio.imu_noise.gyro_sigma", 0.005);
    this->declare_parameter("vio.imu_noise.accel_bias_rw", 0.0005);
    this->declare_parameter("vio.imu_noise.gyro_bias_rw", 0.00005);
}

void VisualOdometryNode::applyCurrentParameters() {
    try {
        window_width_ = this->get_parameter("visualization.window_width").as_int();
        window_height_ = this->get_parameter("visualization.window_height").as_int();
        show_original_ = this->get_parameter("visualization.windows.show_original").as_bool();
        show_features_ = this->get_parameter("visualization.windows.show_features").as_bool();
        show_matches_ = this->get_parameter("visualization.windows.show_matches").as_bool();
        window_pos_x_ = this->get_parameter("visualization.window_pos_x").as_int();
        window_pos_y_ = this->get_parameter("visualization.window_pos_y").as_int();

        input_source_ = this->get_parameter("input.source").as_string();
        if (input_source_ == "zed_sdk") {
            zed_interface_ = std::make_unique<ZEDInterface>();
            int serial = this->get_parameter("input.zed.serial_number").as_int();
            std::string res = this->get_parameter("input.zed.resolution").as_string();
            int fps = this->get_parameter("input.zed.fps").as_int();
            std::string depth_mode = this->get_parameter("input.zed.depth_mode").as_string();

            sl::RESOLUTION resolution = sl::RESOLUTION::HD1080;
            if (res == "HD2K") resolution = sl::RESOLUTION::HD2K;
            else if (res == "HD720") resolution = sl::RESOLUTION::HD720;

            sl::DEPTH_MODE depth = sl::DEPTH_MODE::ULTRA;
            if (depth_mode == "PERFORMANCE") depth = sl::DEPTH_MODE::PERFORMANCE;
            else if (depth_mode == "QUALITY") depth = sl::DEPTH_MODE::QUALITY;

            if (!zed_interface_->connect(serial, resolution, fps, depth)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to connect to ZED camera");
                return;
            }
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
                    "Camera params (ZED): %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                    w, h, camera_params_.fx, camera_params_.fy,
                    camera_params_.cx, camera_params_.cy);
            }
            cv::Mat t_cam_imu;
            if (zed_interface_->getCameraImuTransform(R_cam_imu_, t_cam_imu)) {
                has_cam_imu_tf_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "Camera←IMU rotation:\n  [%+.4f %+.4f %+.4f]\n  [%+.4f %+.4f %+.4f]\n  [%+.4f %+.4f %+.4f]",
                    R_cam_imu_.at<double>(0,0), R_cam_imu_.at<double>(0,1), R_cam_imu_.at<double>(0,2),
                    R_cam_imu_.at<double>(1,0), R_cam_imu_.at<double>(1,1), R_cam_imu_.at<double>(1,2),
                    R_cam_imu_.at<double>(2,0), R_cam_imu_.at<double>(2,1), R_cam_imu_.at<double>(2,2));
                RCLCPP_INFO(this->get_logger(),
                    "Camera←IMU translation: (%.4f, %.4f, %.4f) mm",
                    t_cam_imu.at<double>(0), t_cam_imu.at<double>(1), t_cam_imu.at<double>(2));
            }
        }

        fps_window_size_ = this->get_parameter("fps_window_size").as_int();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "applyCurrentParameters: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult VisualOdometryNode::onParamChange(
    const std::vector<rclcpp::Parameter>& params) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto& p : params) {
        if (p.get_name() == "visualization.window_width") window_width_ = p.as_int();
        else if (p.get_name() == "visualization.window_height") window_height_ = p.as_int();
        else if (p.get_name() == "visualization.windows.show_original") show_original_ = p.as_bool();
        else if (p.get_name() == "visualization.windows.show_features") show_features_ = p.as_bool();
        else if (p.get_name() == "visualization.windows.show_matches") show_matches_ = p.as_bool();
    }
    return result;
}

// ─── Callbacks ──────────────────────────────────────────────────────────────

void VisualOdometryNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (camera_info_received_) return;
    camera_params_.fx = msg->k[0];
    camera_params_.fy = msg->k[4];
    camera_params_.cx = msg->k[2];
    camera_params_.cy = msg->k[5];
    camera_params_.width = msg->width;
    camera_params_.height = msg->height;
    camera_info_received_ = true;

    RCLCPP_INFO(this->get_logger(),
        "Camera params received: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        camera_params_.width, camera_params_.height,
        camera_params_.fx, camera_params_.fy, camera_params_.cx, camera_params_.cy);

    if (!vio_manager_) {
        vio_manager_ = std::make_unique<VIOManager>(
            loadVIOConfig(), this->get_logger(), this->get_clock());
        vio_manager_->setCameraParams(camera_params_);
    }
}

void VisualOdometryNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_source_ == "zed_sdk") return;
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
        cv::Mat rgb = cv_ptr->image;
        cv::Mat depth;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            if (!current_depth_.empty()) depth = current_depth_.clone();
        }
        resource_monitor_->checkResources();
        processImages(rgb, depth);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
    }
}

void VisualOdometryNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_source_ == "zed_sdk") return;
    try {
        auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        std::lock_guard<std::mutex> lock(depth_mutex_);
        cv_ptr->image.copyTo(current_depth_);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
    }
}

void VisualOdometryNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    if (!vio_manager_) return;
    ImuData imu;
    imu.ang_vel_x = msg->angular_velocity.x;
    imu.ang_vel_y = msg->angular_velocity.y;
    imu.ang_vel_z = msg->angular_velocity.z;
    imu.lin_acc_x = msg->linear_acceleration.x;
    imu.lin_acc_y = msg->linear_acceleration.y;
    imu.lin_acc_z = msg->linear_acceleration.z;
    imu.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    imu.valid = true;
    vio_manager_->addImuMeasurement(imu);
}

// ─── ZED SDK ────────────────────────────────────────────────────────────────

void VisualOdometryNode::imuPollLoop() {
    while (rclcpp::ok() && !should_exit_) {
        if (!zed_interface_ || !zed_interface_->isConnected()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        sl::SensorsData sensors;
        if (zed_interface_->getSensorsDataCurrent(sensors) && sensors.imu.is_available) {
            if (vio_manager_) {
                double gx = sensors.imu.angular_velocity.x * M_PI / 180.0;
                double gy = sensors.imu.angular_velocity.y * M_PI / 180.0;
                double gz = sensors.imu.angular_velocity.z * M_PI / 180.0;
                double ax = sensors.imu.linear_acceleration.x;
                double ay = sensors.imu.linear_acceleration.y;
                double az = sensors.imu.linear_acceleration.z;

                ImuData imu;
                if (has_cam_imu_tf_) {
                    const auto& R = R_cam_imu_;
                    imu.ang_vel_x = R.at<double>(0,0)*gx + R.at<double>(0,1)*gy + R.at<double>(0,2)*gz;
                    imu.ang_vel_y = R.at<double>(1,0)*gx + R.at<double>(1,1)*gy + R.at<double>(1,2)*gz;
                    imu.ang_vel_z = R.at<double>(2,0)*gx + R.at<double>(2,1)*gy + R.at<double>(2,2)*gz;
                    imu.lin_acc_x = R.at<double>(0,0)*ax + R.at<double>(0,1)*ay + R.at<double>(0,2)*az;
                    imu.lin_acc_y = R.at<double>(1,0)*ax + R.at<double>(1,1)*ay + R.at<double>(1,2)*az;
                    imu.lin_acc_z = R.at<double>(2,0)*ax + R.at<double>(2,1)*ay + R.at<double>(2,2)*az;
                } else {
                    imu.ang_vel_x = gx; imu.ang_vel_y = gy; imu.ang_vel_z = gz;
                    imu.lin_acc_x = ax; imu.lin_acc_y = ay; imu.lin_acc_z = az;
                }
                imu.timestamp = this->now().seconds();
                imu.valid = true;
                vio_manager_->addImuMeasurement(imu);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

bool VisualOdometryNode::getImages(cv::Mat& rgb, cv::Mat& depth) {
    if (input_source_ == "zed_sdk" && zed_interface_ && zed_interface_->isConnected())
        return zed_interface_->getImages(rgb, depth);
    return false;
}

void VisualOdometryNode::zedTimerCallback() {
    cv::Mat rgb, depth;
    if (getImages(rgb, depth)) {
        std::lock_guard<std::mutex> lock(image_queue_mutex_);
        image_queue_.push({rgb.clone(), depth.clone()});
        if (image_queue_.size() > 2) image_queue_.pop();
        image_ready_.notify_one();
    }
}

VIOConfig VisualOdometryNode::loadVIOConfig() {
    VIOConfig c;
    c.init_mode = this->get_parameter("vio.init_mode").as_string();
    c.use_imu = this->get_parameter("vio.use_imu").as_bool();
    c.graph_window_size = this->get_parameter("vio.graph_window_size").as_int();
    c.lost_timeout_sec = this->get_parameter("vio.lost_timeout_sec").as_double();
    c.vocabulary_path = this->get_parameter("vio.vocabulary_path").as_string();
    c.loop_closure_enable = this->get_parameter("vio.loop_closure.enable").as_bool();
    c.loop_min_matches = this->get_parameter("vio.loop_closure.min_matches").as_int();
    c.loop_min_score = this->get_parameter("vio.loop_closure.min_score").as_double();
    c.loop_min_interval_keyframes = this->get_parameter("vio.loop_closure.min_interval_keyframes").as_int();
    c.bayesian_threshold = this->get_parameter("vio.loop_closure.bayesian_threshold").as_double();
    c.bayesian_temporal_consistency = this->get_parameter("vio.loop_closure.temporal_consistency").as_int();
    c.stm_size = this->get_parameter("vio.memory.stm_size").as_int();
    c.rehearsal_similarity = this->get_parameter("vio.memory.rehearsal_similarity").as_double();
    c.proximity_max_dist = this->get_parameter("vio.memory.proximity_max_dist").as_double();
    c.imu_params.accel_noise_sigma = this->get_parameter("vio.imu_noise.accel_sigma").as_double();
    c.imu_params.gyro_noise_sigma = this->get_parameter("vio.imu_noise.gyro_sigma").as_double();
    c.imu_params.accel_bias_rw_sigma = this->get_parameter("vio.imu_noise.accel_bias_rw").as_double();
    c.imu_params.gyro_bias_rw_sigma = this->get_parameter("vio.imu_noise.gyro_bias_rw").as_double();
    return c;
}

// ─── Core Processing ────────────────────────────────────────────────────────

void VisualOdometryNode::processImages(const cv::Mat& rgb, const cv::Mat& depth) {
    if (!vio_manager_) return;

    auto start = std::chrono::steady_clock::now();
    double ts = this->now().seconds();
    auto output = vio_manager_->processFrame(rgb, depth, ts);
    auto end = std::chrono::steady_clock::now();
    double proc_ms = std::chrono::duration<double, std::milli>(end - start).count();

    frame_count_++;
    double now_s = std::chrono::duration<double>(end.time_since_epoch()).count();
    if (fps_start_time_ == 0.0) fps_start_time_ = now_s;
    double elapsed = now_s - fps_start_time_;
    if (elapsed >= 1.0) {
        current_fps_ = frame_count_ / elapsed;
        frame_count_ = 0;
        fps_start_time_ = now_s;
    }

    if (visualizer_ && vio_manager_) {
        auto vis = vio_manager_->getVisualizationData();
        Features feat;
        feat.keypoints = vis.features;
        FeatureMatches matches;
        matches.prev_points = vis.matched_ref;
        matches.curr_points = vis.matched_cur;
        visualizer_->visualize(rgb, feat, matches, prev_frame_);
    }

    if (!output.T_world_body.empty()) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "[%.1ffps %.0fms] Pose: (%.3f, %.3f, %.3f)m  state=%s  q=%.2f  feat=%d  inl=%d",
            current_fps_, proc_ms,
            output.T_world_body.at<double>(0,3),
            output.T_world_body.at<double>(1,3),
            output.T_world_body.at<double>(2,3),
            vo::vioStateStr(output.state),
            output.tracking_quality,
            output.num_tracked_features,
            output.num_map_matches);
    }

    // Publish
    if (this->get_parameter("processing.publish_results").as_bool() &&
        this->get_parameter("processing.enable_pose_estimation").as_bool()) {
        publishResults(output, proc_ms);
    }

    resource_monitor_->updateProcessingTime(proc_ms);
    prev_frame_ = rgb.clone();
}

void VisualOdometryNode::publishResults(const VIOOutput& output, double processing_time_ms) {
    auto stamp = this->now();
    std::string frame_id = this->get_parameter("frames.frame_id").as_string();

    cv::Mat T = output.T_world_body;
    if (T.empty()) return;
    double px = T.at<double>(0,3);
    double py = T.at<double>(1,3);
    double pz = T.at<double>(2,3);

    cv::Mat R = T(cv::Rect(0,0,3,3));
    tf2::Matrix3x3 R_tf(
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2));
    tf2::Quaternion q;
    R_tf.getRotation(q);

    // PoseStamped
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_id;
    pose_msg.pose.position.x = px;
    pose_msg.pose.position.y = py;
    pose_msg.pose.position.z = pz;
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    pose_pub_->publish(pose_msg);

    // TF
    if (this->get_parameter("tf.publish").as_bool()) {
        std::string child = this->get_parameter("frames.child_frame_id").as_string();
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = frame_id;
        tf_msg.child_frame_id = child;
        tf_msg.transform.translation.x = px;
        tf_msg.transform.translation.y = py;
        tf_msg.transform.translation.z = pz;
        tf_msg.transform.rotation = pose_msg.pose.orientation;
        tf_broadcaster_->sendTransform(tf_msg);
    }

    // VOState
    visual_odometry::msg::VOState vo_msg;
    vo_msg.header.stamp = stamp;
    vo_msg.header.frame_id = frame_id;
    vo_msg.pose = pose_msg.pose;
    vo_msg.num_features = static_cast<uint32_t>(output.num_tracked_features);
    vo_msg.num_matches = static_cast<uint32_t>(output.num_map_matches);
    vo_msg.tracking_quality = static_cast<float>(output.tracking_quality);
    vo_msg.scale_confidence = 1.0f;
    vo_msg.processing_time = static_cast<float>(processing_time_ms);
    vo_state_pub_->publish(vo_msg);
}

void VisualOdometryNode::publishStaticTransform() {
    geometry_msgs::msg::TransformStamped st;
    st.header.stamp = this->now();
    st.header.frame_id = "camera_link";
    st.child_frame_id = "camera_optical_frame";
    st.transform.translation.x = 0.0;
    st.transform.translation.y = 0.0;
    st.transform.translation.z = 0.0;
    tf2::Matrix3x3 R(0, 0, 1, -1, 0, 0, 0, -1, 0);
    tf2::Quaternion q;
    R.getRotation(q);
    st.transform.rotation.x = q.x();
    st.transform.rotation.y = q.y();
    st.transform.rotation.z = q.z();
    st.transform.rotation.w = q.w();
    static_tf_broadcaster_->sendTransform(st);
}

void VisualOdometryNode::processingLoop() {
    while (rclcpp::ok() && !should_exit_) {
        std::unique_lock<std::mutex> lock(image_queue_mutex_);
        image_ready_.wait(lock, [this]() { return !image_queue_.empty() || should_exit_; });
        if (should_exit_) break;
        auto [rgb, depth] = image_queue_.front();
        image_queue_.pop();
        lock.unlock();
        processImages(rgb, depth);
        resource_monitor_->checkResources();
    }
}

}  // namespace vo
