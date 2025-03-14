#include "visual_odometry/logger.hpp"
#include <sstream>
#include <iomanip>

namespace vo {

Logger::Logger(rclcpp::Node* node) : node_(node) {}

void Logger::logSystemInfo(const SystemInfo& info) {
    std::stringstream ss;
    ss << "\n=== Visual Odometry System Information ==="
       << "\n[Window Settings]"
       << "\n- Size: " << info.window_width << "x" << info.window_height
       << "\n- Position: (" << info.window_pos_x << ", " << info.window_pos_y << ")"
       << "\n[Display Options]"
       << "\n- Show Original: " << (info.show_original ? "Enabled" : "Disabled")
       << "\n- Show Features: " << (info.show_features ? "Enabled" : "Disabled")
       << "\n- Show Matches: " << (info.show_matches ? "Enabled" : "Disabled")
       << "\n[Feature Detection]"
       << "\n- Max Features: " << info.max_features
       << "\n- Scale Factor: " << std::fixed << std::setprecision(2) << info.scale_factor
       << "\n- Pyramid Levels: " << info.n_levels
       << "\n[Input Source]"
       << "\n- Mode: " << info.input_source
       << "\n[Topics]"
       << "\n- RGB: " << info.rgb_topic
       << "\n- Depth: " << info.depth_topic
       << "\n- Camera Info: " << info.camera_info_topic
       << "\n[Processing]"
       << "\n- Queue Size: " << info.queue_size
       << "\n- FPS Target: " << info.target_fps
       << "\n[Camera Parameters]"
       << "\n- Resolution: " << info.camera_width << "x" << info.camera_height
       << "\n- Focal Length: (" << std::fixed << std::setprecision(2) 
       << info.camera_fx << ", " << info.camera_fy << ")"
       << "\n- Principal Point: (" << std::fixed << std::setprecision(2) 
       << info.camera_cx << ", " << info.camera_cy << ")"
       << "\n[System]"
       << "\n- OpenCV Version: " << info.opencv_version
       << "\n- CUDA Available: " << (info.cuda_available ? "Yes" : "No")
       << "\n- Build Type: " << info.build_type
       << "\n================================";

    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
}

void Logger::logPerformance(const PerformanceMetrics& metrics) {
    std::stringstream ss;
    ss << "\n[Processing Performance]"
       << "\n- Feature Detection: " << std::fixed << std::setprecision(1) 
       << metrics.detection_time << " ms (" 
       << 1000.0/metrics.detection_time << " FPS)"
       << "\n- Feature Matching: " << metrics.matching_time << " ms"
       << "\n- Visualization: " << metrics.visualization_time << " ms"
       << "\n[Detection Results]"
       << "\n- Features: " << metrics.num_features
       << "\n- Matches: " << metrics.num_matches
       << "\n- Matching Ratio: " << std::fixed << std::setprecision(2) 
       << metrics.matching_ratio
       << "\n- Average Movement: " << metrics.avg_movement << " px"
       << "\n[Resource Monitor]"
       << "\n- Memory Usage: " << std::fixed << std::setprecision(2) 
       << metrics.memory_usage << " MB"
       << "\n- Queue Size: " << metrics.queue_size
       << "\n- Processing Time: " << metrics.processing_time << " ms";

    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
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

} // namespace vo 