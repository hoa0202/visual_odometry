#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class FeatureMatcher {
public:
    FeatureMatcher();
    
    FeatureMatches match(const Features& prev_features, 
                        const Features& curr_features,
                        const cv::Mat& prev_frame_gray,
                        const cv::Mat& curr_frame_gray);

private:
    FeatureMatches matchOpticalFlow(const Features& prev_features,
                                  const Features& curr_features,
                                  const cv::Mat& prev_frame_gray,
                                  const cv::Mat& curr_frame_gray);
                                  
    FeatureMatches matchDescriptors(const Features& prev_features,
                                  const Features& curr_features);

    // 매칭 관련 멤버 변수
    cv::Ptr<cv::BFMatcher> matcher_;
    std::vector<cv::Point2f> prev_pts_;
    std::vector<cv::Point2f> curr_pts_;
};

} // namespace vo 