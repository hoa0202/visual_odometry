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

    // 매칭 관련 함수 추가
    FeatureMatches matchFeatures(const Features& prev_features, 
                               const Features& curr_features);
    
    // 매칭 파라미터 설정
    void setMatchingParams(float ratio_threshold = 0.7f) {
        ratio_threshold_ = ratio_threshold;
    }

private:
    // detector 업데이트 메서드
    void updateDetector();

    cv::Ptr<cv::ORB> detector_;
    int max_features_ = 2000;
    float scale_factor_ = 1.2f;
    int n_levels_ = 8;
    int visualization_flags_ = static_cast<int>(cv::DrawMatchesFlags::DEFAULT);
    bool visualization_needed_ = true;

    // 매칭 관련 멤버
    cv::Ptr<cv::DescriptorMatcher> matcher_{cv::DescriptorMatcher::create("BruteForce-Hamming")};
    float ratio_threshold_{0.7f};

    // FLANN 관련 멤버 제거
    // cv::FlannBasedMatcher flann_matcher_;
    // bool use_flann_{true};

    // FLANN 매칭을 위한 버퍼 추가
    // cv::Mat prev_descriptors_32f_;
    // cv::Mat curr_descriptors_32f_;
};

} // namespace vo 