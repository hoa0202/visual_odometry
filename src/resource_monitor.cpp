#include "visual_odometry/resource_monitor.hpp"
#include <sys/resource.h>

namespace vo {

ResourceMonitor::ResourceMonitor(rclcpp::Node* node) 
    : node_(node),
      last_check_time_(std::chrono::steady_clock::now()) {
}

void ResourceMonitor::updateProcessingTime(double time_ms) {
    std::lock_guard<std::mutex> lock(metric_mutex_);
    current_metrics_.processing_time = time_ms;
}

void ResourceMonitor::updateQueueSize(size_t size) {
    std::lock_guard<std::mutex> lock(metric_mutex_);
    current_metrics_.queue_size = size;
}

void ResourceMonitor::checkResources() {
    auto current_time = std::chrono::steady_clock::now();
    
    if (std::chrono::duration_cast<std::chrono::seconds>(
        current_time - last_check_time_).count() >= CHECK_INTERVAL_SEC) {
        
        std::lock_guard<std::mutex> lock(metric_mutex_);
        
        // 시스템 리소스 사용량 체크
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            current_metrics_.memory_usage = usage.ru_maxrss / 1024.0;  // MB
        }
        
        // 메트릭 히스토리 업데이트
        metrics_history_.push_back(current_metrics_);
        if (metrics_history_.size() > HISTORY_SIZE) {
            metrics_history_.pop_front();
        }
        
        last_check_time_ = current_time;
    }
}

const ResourceMonitor::ResourceMetrics& ResourceMonitor::getCurrentMetrics() const {
    return current_metrics_;
}

} // namespace vo 