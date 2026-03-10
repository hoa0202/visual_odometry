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
        // PnP 결과 (curr→prev)
        bool pnp_success{false};
        int pnp_inliers{0};
        cv::Mat R;  // 3x3
        cv::Mat t;  // 3x1
    };

    ProcessingResult processFrame(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const CameraParams& camera_params,
                                bool first_frame = false,
                                bool enable_pose_estimation = true);

    void setPreviousFrame(const cv::Mat& frame, const cv::Mat& depth,
                         const Features& features);

private:
    // 컴포넌트들
    std::shared_ptr<FeatureDetector> feature_detector_;
    std::shared_ptr<FeatureMatcher> feature_matcher_;

    cv::Mat prev_frame_gray_;
    cv::Mat prev_depth_;
    Features prev_features_;

    void backprojectAndFilter(FeatureMatches& matches,
                              const cv::Mat& depth,
                              const CameraParams& camera_params,
                              bool use_curr_points = false);  // true: curr 2D+depth → 3D (prev_depth NaN 대비)
    cv::Mat preprocessFrame(const cv::Mat& rgb);
    Features detectFeatures(const cv::Mat& gray);
    FeatureMatches matchFeatures(const Features& curr_features, 
                               const cv::Mat& curr_gray);
};

} // namespace vo 