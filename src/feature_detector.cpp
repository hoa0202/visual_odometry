#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FeatureDetector::FeatureDetector() {
    // 초기 detector 생성
    updateDetector();
}

Features FeatureDetector::detectFeatures(const cv::Mat& image) {
    Features features;
    
    detector_->detectAndCompute(
        image, 
        cv::Mat(), 
        features.keypoints,
        features.descriptors
    );

    features.visualization = image.clone();
    if (image.channels() == 1) {
        cv::cvtColor(image, features.visualization, cv::COLOR_GRAY2BGR);
    }
    
    cv::drawKeypoints(
        features.visualization,
        features.keypoints,
        features.visualization,
        cv::Scalar(0, 255, 0),
        static_cast<cv::DrawMatchesFlags>(visualization_flags_)
    );

    return features;
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

void FeatureDetector::setVisualizationFlags(int flags) {
    visualization_flags_ = flags;
}

void FeatureDetector::updateDetector() {
    detector_ = cv::ORB::create(
        max_features_,     // 최대 특징점 수
        scale_factor_,     // 스케일 팩터
        n_levels_          // 피라미드 레벨 수
    );
}

} // namespace vo 