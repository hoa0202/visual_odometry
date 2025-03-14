#include "visual_odometry/logger.hpp"
#include <sstream>
#include <iomanip>
#include <deque>

namespace vo {

Logger::Logger(rclcpp::Node* node) : node_(node),
    last_fps_update_(node->now()),
    current_fps_(0.0),
    avg_latency_(0.0) {
    if (!node_) {
        throw std::runtime_error("Null node pointer in Logger constructor");
    }
    // 로그 레벨 설정
    auto logger = node_->get_logger();
    rcutils_ret_t ret = rcutils_logging_set_logger_level(
        logger.get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    if (ret != RCUTILS_RET_OK) {
        throw std::runtime_error("Failed to set logger level");
    }
}

void Logger::updateMetrics(const ProcessingMetrics& metrics) {
    auto current_time = node_->now();
    frame_times_.push_back(metrics.processing_time);
    
    if (frame_times_.size() > FPS_WINDOW_SIZE) {
        frame_times_.pop_front();
    }

    // FPS와 지연률 계산 (1초마다)
    double elapsed = (current_time - last_fps_update_).seconds();
    if (elapsed >= FPS_UPDATE_INTERVAL) {
        // FPS 계산
        double total_time = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0);
        avg_latency_ = total_time / frame_times_.size();
        current_fps_ = 1000.0 / avg_latency_;  // ms를 초로 변환

        std::stringstream ss;
        ss << "Performance Metrics:"
           << "\n- Feature Detection: " << std::fixed << std::setprecision(1) 
           << metrics.detection_time << " ms (" 
           << (1000.0 / metrics.detection_time) << " FPS)"
           << "\n- Feature Matching: " << metrics.matching_time << " ms"
           << "\n- Visualization: " << metrics.visualization_time << " ms"
           << "\n- Total Processing: " << avg_latency_ << " ms (" 
           << current_fps_ << " FPS)"
           << "\n- Memory Usage: " << metrics.memory_usage << " MB"
           << "\n- Queue Size: " << metrics.queue_size
           << "\n- Features: " << metrics.num_features
           << "\n- Matches: " << metrics.num_matches 
           << " (" << std::setprecision(1) << (metrics.matching_ratio * 100) << "%)";

        RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
        last_fps_update_ = current_time;
    }
}

void Logger::logSystemInfo(const SystemInfo& info) {
    std::stringstream ss;
    ss << "\n====== Visual Odometry System Configuration ======\n"
       << "\n[Visualization Settings]"
       << "\n- Window Size: " << info.window_width << "x" << info.window_height
       << "\n- Window Position: (" << info.window_pos_x << ", " << info.window_pos_y << ")"
       << "\n- Show Original: " << (info.show_original ? "Yes" : "No")
       << "\n- Show Features: " << (info.show_features ? "Yes" : "No")
       << "\n- Show Matches: " << (info.show_matches ? "Yes" : "No")
       << "\n\n[Feature Detection Parameters]"
       << "\n- Maximum Features: " << info.max_features
       << "\n- Scale Factor: " << std::fixed << std::setprecision(2) << info.scale_factor
       << "\n- Pyramid Levels: " << info.n_levels
       << "\n\n[Input Configuration]"
       << "\n- Source: " << info.input_source
       << "\n- RGB Topic: " << info.rgb_topic
       << "\n- Depth Topic: " << info.depth_topic
       << "\n- Camera Info Topic: " << info.camera_info_topic
       << "\n\n[Processing Settings]"
       << "\n- Queue Size: " << info.queue_size
       << "\n- Target FPS: " << info.target_fps
       << "\n\n[Camera Parameters]"
       << "\n- Resolution: " << info.camera_width << "x" << info.camera_height
       << "\n- Focal Length: (fx=" << info.camera_fx << ", fy=" << info.camera_fy << ")"
       << "\n- Principal Point: (cx=" << info.camera_cx << ", cy=" << info.camera_cy << ")"
       << "\n\n============================================";
    
    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
}

void Logger::logProcessingMetrics(const ProcessingMetrics& metrics) {
    // This function is now empty as per the new implementation
}

void Logger::logError(const std::string& component, const std::string& message) {
    RCLCPP_ERROR(node_->get_logger(), "[%s] %s", component.c_str(), message.c_str());
}

void Logger::logWarning(const std::string& component, const std::string& message) {
    RCLCPP_WARN(node_->get_logger(), "[%s] %s", component.c_str(), message.c_str());
}

void Logger::logDebug(const std::string& component, const std::string& message) {
    RCLCPP_DEBUG(node_->get_logger(), "[%s] %s", component.c_str(), message.c_str());
}

void Logger::logInfo(const std::string& component, const std::string& message) {
    RCLCPP_INFO(node_->get_logger(), "[%s] %s", component.c_str(), message.c_str());
}

void Logger::logParameterUpdate(const std::string& param_name, const rclcpp::Parameter& param) {
    std::stringstream ss;
    ss << "Parameter updated - " << param_name << ": ";
    
    switch (param.get_type()) {
        case rclcpp::ParameterType::PARAMETER_BOOL:
            ss << (param.as_bool() ? "true" : "false");
            break;
        case rclcpp::ParameterType::PARAMETER_INTEGER:
            ss << param.as_int();
            break;
        case rclcpp::ParameterType::PARAMETER_DOUBLE:
            ss << std::fixed << std::setprecision(2) << param.as_double();
            break;
        case rclcpp::ParameterType::PARAMETER_STRING:
            ss << param.as_string();
            break;
        default:
            ss << "[unsupported type]";
    }
    
    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
}

void Logger::logVisualizationParameters(int window_width, int window_height,
                                     int window_pos_x, int window_pos_y,
                                     bool show_original, bool show_features,
                                     bool show_matches) {
    std::stringstream ss;
    ss << "Visualization Parameters:"
       << "\n  - window_size: " << window_width << "x" << window_height
       << "\n  - window_position: (" << window_pos_x << ", " << window_pos_y << ")"
       << "\n  - show_original: " << (show_original ? "true" : "false")
       << "\n  - show_features: " << (show_features ? "true" : "false")
       << "\n  - show_matches: " << (show_matches ? "true" : "false");

    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
}

void Logger::logFeatureDetectorParameters(int max_features, double scale_factor,
                                       int n_levels, const std::string& viz_type) {
    std::stringstream ss;
    ss << "Feature Detector Parameters:"
       << "\n  - max_features: " << max_features
       << "\n  - scale_factor: " << std::fixed << std::setprecision(2) << scale_factor
       << "\n  - n_levels: " << n_levels
       << "\n  - visualization_type: " << viz_type;

    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
}

} // namespace vo 