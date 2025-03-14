#pragma once

#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <mutex>

namespace vo {

class ResourceMonitor {
public:
    struct ResourceMetrics {
        double memory_usage{0.0};  // MB
        double processing_time{0.0};  // ms
        size_t queue_size{0};
    };

    explicit ResourceMonitor(rclcpp::Node* node);

    void updateProcessingTime(double time_ms);
    void updateQueueSize(size_t size);
    void checkResources();
    double getMemoryUsage() const;
    const ResourceMetrics& getCurrentMetrics() const;

private:
    static constexpr size_t HISTORY_SIZE = 100;
    static constexpr int CHECK_INTERVAL_SEC = 5;

    rclcpp::Node* node_;
    std::chrono::steady_clock::time_point last_check_time_;
    ResourceMetrics current_metrics_;
    std::deque<ResourceMetrics> metrics_history_;
    mutable std::mutex metric_mutex_;
};

} // namespace vo 