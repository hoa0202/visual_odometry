#include "visual_odometry/resource_monitor.hpp"
#include <sys/resource.h>
#include <unistd.h>

namespace vo {

ResourceMonitor::ResourceMonitor(rclcpp::Node* node) : node_(node) {
    if (!node_) {
        throw std::runtime_error("Null node pointer in ResourceMonitor constructor");
    }
    last_check_time_ = std::chrono::steady_clock::now();
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
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_check_time_).count();
    
    if (elapsed >= CHECK_INTERVAL_SEC) {
        // 메모리 사용량 업데이트
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            std::lock_guard<std::mutex> lock(metric_mutex_);
            // Linux에서는 KB 단위로 반환되므로 MB로 변환
            current_metrics_.memory_usage = static_cast<double>(usage.ru_maxrss) / 1024.0;
        }
        
        last_check_time_ = now;
        
        // 메트릭 히스토리 업데이트
        metrics_history_.push_back(current_metrics_);
        if (metrics_history_.size() > HISTORY_SIZE) {
            metrics_history_.pop_front();
        }
    }
}

const ResourceMonitor::ResourceMetrics& ResourceMonitor::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metric_mutex_);
    return current_metrics_;
}

double ResourceMonitor::getMemoryUsage() const {
    std::lock_guard<std::mutex> lock(metric_mutex_);
    return current_metrics_.memory_usage;
}

} // namespace vo 