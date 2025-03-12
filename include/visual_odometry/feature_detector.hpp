#pragma once

#include <opencv2/features2d.hpp>
#include "visual_odometry/types.hpp"
#include <memory>

namespace vo {

class FeatureDetector {
public:
    FeatureDetector();
    ~FeatureDetector() = default;

    // 특징점 검출 메서드
    Features detectFeatures(const cv::Mat& frame, int max_features, int fast_threshold);
    
    // 특징점 매칭 메서드
    FeatureMatches matchFeatures(const Features& prev_features, const Features& curr_features);

    // 파라미터 설정 메서드들
    void setMaxFeatures(int max_features) { max_features_ = max_features; updateDetector(); }
    void setScaleFactor(double scale_factor) { scale_factor_ = scale_factor; updateDetector(); }
    void setNLevels(int n_levels) { n_levels_ = n_levels; updateDetector(); }
    void setVisualizationType(const std::string& type);

private:
    // 특징점 검출기와 디스크립터 추출기
    cv::Ptr<cv::FastFeatureDetector> detector_;
    cv::Ptr<cv::ORB> descriptor_;
    cv::Ptr<cv::BFMatcher> matcher_;

    // 파라미터들
    int max_features_{500};
    double scale_factor_{1.2};
    int n_levels_{8};
    int visualization_flags_{static_cast<int>(cv::DrawMatchesFlags::DEFAULT)};

    // 버퍼 재사용
    cv::Mat descriptors_buffer_;
    std::vector<cv::KeyPoint> keypoints_buffer_;

    // 내부 메서드
    void updateDetector();
};

} // namespace vo 