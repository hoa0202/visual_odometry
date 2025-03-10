#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FeatureDetector::FeatureDetector() {
    // 초기 detector 생성
    updateDetector();
}

Features FeatureDetector::detectFeatures(const cv::Mat& gray) {
    Features features;
    
    // FAST 파라미터 최적화
    static const int fast_threshold = 20;  // static으로 변경
    static const bool non_max_suppression = true;
    
    // 메모리 재사용
    static std::vector<cv::KeyPoint> keypoints;
    keypoints.clear();  // 벡터 재사용
    
    cv::FAST(gray, keypoints, fast_threshold, non_max_suppression);
    
    // 특징점 개수 제한 (부호 없는 정수 비교 수정)
    if (static_cast<int>(keypoints.size()) > max_features_) {
        std::partial_sort(keypoints.begin(), 
                         keypoints.begin() + max_features_,
                         keypoints.end(),
                         [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                             return a.response > b.response;
                         });
        keypoints.resize(max_features_);
    }
    
    features.keypoints = keypoints;
    
    // 시각화는 필요한 경우만
    if (visualization_needed_) {
        cv::Mat vis = cv::Mat::zeros(gray.size(), CV_8UC3);
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        for (const auto& kp : keypoints) {
            cv::circle(vis, kp.pt, 3, cv::Scalar(0,255,0), -1);
        }
        features.visualization = vis;
    }
    
    return features;
}

void FeatureDetector::setMaxFeatures(int max_features) {
    detector_ = cv::ORB::create(
        max_features,     // nfeatures
        1.2f,            // scaleFactor
        8,               // nlevels
        31,             // edgeThreshold
        0,              // firstLevel
        2,              // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,             // patchSize
        20              // fastThreshold
    );
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