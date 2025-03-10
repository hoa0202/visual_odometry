#pragma once

#include <opencv2/features2d.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class FeatureDetector {
public:
    explicit FeatureDetector();
    ~FeatureDetector() = default;

    // 특징점 검출 메서드
    Features detectFeatures(const cv::Mat& image);
    
    // 파라미터 설정 메서드들
    void setMaxFeatures(int max_features);
    void setScaleFactor(float scale_factor);
    void setNLevels(int n_levels);
    void setVisualizationType(const std::string& type);

private:
    // detector 업데이트 메서드
    void updateDetector();

    cv::Ptr<cv::Feature2D> detector_;
    int max_features_{10000};
    float scale_factor_{1.2f};
    int n_levels_{8};
    int visualization_flags_{static_cast<int>(cv::DrawMatchesFlags::DEFAULT)};
};

} // namespace vo 