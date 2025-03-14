#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <sys/resource.h>
#include <deque>

namespace vo {

class ResourceMonitor {
public:
    struct ResourceMetrics {
        double memory_usage;
        size_t queue_size;
        double processing_time;
        
        ResourceMetrics() 
            : memory_usage(0.0), queue_size(0), processing_time(0.0) {}
    };

    explicit ResourceMonitor(rclcpp::Node* node);
    ~ResourceMonitor() = default;

    void updateQueueSize(size_t size);
    void updateProcessingTime(double time);
    void checkResources();
    const ResourceMetrics& getCurrentMetrics() const;

private:
    rclcpp::Node* node_;
    std::chrono::steady_clock::time_point last_check_time_;
    
    // 모니터링 메트릭
    double processing_time_{0.0};
    size_t queue_size_{0};
    
    std::mutex metric_mutex_;
    
    ResourceMetrics current_metrics_;
    
    // 모니터링 설정
    const int CHECK_INTERVAL_SEC = 5;  // 5초마다 체크
    const size_t HISTORY_SIZE = 10;    // 메트릭 히스토리 크기
    std::deque<ResourceMetrics> metrics_history_;
};

} // namespace vo 