#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <deque>

namespace vo {

struct SystemInfo {
    // 윈도우 설정
    int window_width;
    int window_height;
    int window_pos_x;
    int window_pos_y;
    bool show_original;
    bool show_features;
    bool show_matches;

    // 특징점 검출 설정
    int max_features;
    float scale_factor;
    int n_levels;

    // 입력 소스 설정
    std::string input_source;
    std::string rgb_topic;
    std::string depth_topic;
    std::string camera_info_topic;

    // 처리 설정
    int queue_size;
    double target_fps;

    // 카메라 파라미터
    int camera_width;
    int camera_height;
    double camera_fx;
    double camera_fy;
    double camera_cx;
    double camera_cy;
};

struct ProcessingMetrics {
    double detection_time{0.0};
    double matching_time{0.0};
    double visualization_time{0.0};
    int num_features{0};
    int num_matches{0};
    double matching_ratio{0.0};
    double avg_movement{0.0};
    double memory_usage{0.0};
    size_t queue_size{0};
    double processing_time{0.0};
};

class Logger {
public:
    explicit Logger(rclcpp::Node* node);

    void updateMetrics(const ProcessingMetrics& metrics);
    void logSystemInfo(const SystemInfo& info);
    void logProcessingMetrics(const ProcessingMetrics& metrics);
    void logError(const std::string& component, const std::string& message);
    void logWarning(const std::string& component, const std::string& message);
    void logDebug(const std::string& component, const std::string& message);
    void logInfo(const std::string& component, const std::string& message);
    void logParameterUpdate(const std::string& param_name, const rclcpp::Parameter& param);
    void logVisualizationParameters(int window_width, int window_height,
                                  int window_pos_x, int window_pos_y,
                                  bool show_original, bool show_features,
                                  bool show_matches);
    void logFeatureDetectorParameters(int max_features, double scale_factor,
                                    int n_levels, const std::string& viz_type);

private:
    rclcpp::Node* node_;
    static constexpr size_t FPS_WINDOW_SIZE = 30;
    static constexpr double FPS_UPDATE_INTERVAL = 1.0;
    std::deque<double> frame_times_;
    rclcpp::Time last_fps_update_;
    double current_fps_;
    double avg_latency_;
};

} // namespace vo 