#pragma once

#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/feature_matcher.hpp"
#include <rclcpp/logger.hpp>

namespace vo {

class FrameProcessor {
public:
    FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                  std::shared_ptr<FeatureMatcher> matcher);

    struct ProcessingResult {
        Features features;
        FeatureMatches matches;
        double feature_detection_time{0.0};
        double feature_matching_time{0.0};
        double visualization_time{0.0};
        bool is_keyframe{false};
    };

    ProcessingResult processFrame(const cv::Mat& rgb, 
                                const cv::Mat& depth,
                                bool first_frame = false);

    // 이전 프레임 정보 설정
    void setPreviousFrame(const cv::Mat& frame, const Features& features);

private:
    // 컴포넌트들
    std::shared_ptr<FeatureDetector> feature_detector_;
    std::shared_ptr<FeatureMatcher> feature_matcher_;

    // 이전 프레임 데이터
    cv::Mat prev_frame_gray_;
    Features prev_features_;

    // 내부 처리 메서드들
    cv::Mat preprocessFrame(const cv::Mat& rgb);
    Features detectFeatures(const cv::Mat& gray);
    FeatureMatches matchFeatures(const Features& curr_features, 
                               const cv::Mat& curr_gray);
};

} // namespace vo 