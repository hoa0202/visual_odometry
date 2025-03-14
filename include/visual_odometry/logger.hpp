#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>

namespace vo {

struct SystemInfo {
    // Window Settings
    int window_width{0};
    int window_height{0};
    int window_pos_x{0};
    int window_pos_y{0};
    
    // Display Options
    bool show_original{false};
    bool show_features{false};
    bool show_matches{false};
    
    // Feature Detection
    int max_features{0};
    float scale_factor{0.0f};
    int n_levels{0};
    
    // Input Source
    std::string input_source;
    
    // Topics
    std::string rgb_topic;
    std::string depth_topic;
    std::string camera_info_topic;
    
    // Processing
    size_t queue_size{0};
    int target_fps{0};
    
    // Camera Parameters
    int camera_width{0};
    int camera_height{0};
    double camera_fx{0.0};
    double camera_fy{0.0};
    double camera_cx{0.0};
    double camera_cy{0.0};
    
    // System
    std::string opencv_version;
    bool cuda_available{false};
    std::string build_type;
};

struct PerformanceMetrics {
    double detection_time{0.0};
    double matching_time{0.0};
    double visualization_time{0.0};
    size_t num_features{0};
    size_t num_matches{0};
    float matching_ratio{0.0f};
    float avg_movement{0.0f};
    double memory_usage{0.0};
    size_t queue_size{0};
    double processing_time{0.0};
};

class Logger {
public:
    explicit Logger(rclcpp::Node* node);

    void logSystemInfo(const SystemInfo& info);
    void logPerformance(const PerformanceMetrics& metrics);
    void logError(const std::string& component, const std::string& message);
    void logWarning(const std::string& component, const std::string& message);
    void logDebug(const std::string& component, const std::string& message);
    void logInfo(const std::string& component, const std::string& message);

private:
    rclcpp::Node* node_;
};

} // namespace vo 