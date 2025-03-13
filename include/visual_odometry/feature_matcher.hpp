#pragma once

#include <opencv2/features2d.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class FeatureMatcher {
public:
    FeatureMatcher();
    
    // 특징점 매칭을 수행하는 단일 함수
    FeatureMatches match(const Features& prev_features,
                        const Features& curr_features,
                        const cv::Mat& prev_frame_gray,
                        const cv::Mat& curr_frame_gray);

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_threshold_;  // Lowe's ratio test threshold
};

} // namespace vo 