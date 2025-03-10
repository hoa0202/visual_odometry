#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FeatureDetector::FeatureDetector() {
    // 초기 detector 생성
    updateDetector();
}

Features FeatureDetector::detectFeatures(const cv::Mat& gray_image) {
    Features result;
    
    // 특징점 검출
    detector_->detect(gray_image, result.keypoints);
    
    // 최대 특징점 수 제한
    if (static_cast<int>(result.keypoints.size()) > max_features_) {
        result.keypoints.resize(max_features_);
    }
    
    // 시각화
    result.visualization = gray_image.clone();
    if (gray_image.channels() == 1) {
        cv::cvtColor(gray_image, result.visualization, cv::COLOR_GRAY2BGR);
    }
    
    cv::drawKeypoints(result.visualization, 
                     result.keypoints, 
                     result.visualization,
                     cv::Scalar(0, 255, 0),  // 녹색
                     static_cast<cv::DrawMatchesFlags>(visualization_flags_));
    
    return result;
}

void FeatureDetector::setMaxFeatures(int max_features) {
    if (max_features_ != max_features) {
        max_features_ = max_features;
        updateDetector();
        RCLCPP_INFO(rclcpp::get_logger("feature_detector"), 
                    "Max features updated to: %d", max_features_);
    }
}

void FeatureDetector::setScaleFactor(float scale_factor) {
    scale_factor_ = scale_factor;
    updateDetector();
}

void FeatureDetector::setNLevels(int n_levels) {
    n_levels_ = n_levels;
    updateDetector();
}

void FeatureDetector::setVisualizationType(const std::string& type) {
    visualization_flags_ = static_cast<int>(
        type == "rich" ? 
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS : 
        cv::DrawMatchesFlags::DEFAULT);
}

void FeatureDetector::updateDetector() {
    detector_ = cv::ORB::create(
        max_features_,     // 최대 특징점 수
        scale_factor_,     // 스케일 팩터
        n_levels_          // 피라미드 레벨 수
    );
}

} // namespace vo 