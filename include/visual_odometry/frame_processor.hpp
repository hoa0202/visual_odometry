#pragma once

#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/feature_matcher.hpp"
#include <rclcpp/logger.hpp>

namespace vo {

/** IMU-predicted relative pose (optical frame, T_prev_from_curr). Phase 4: feature filtering용. */
struct ImuPredictedPose {
    cv::Mat R;      // 3x3 rotation (optical frame)
    cv::Mat t;      // 3x1 translation (optical frame, mm) — rotation-only 모드에서 zero
    double angular_rate{0.0};  // rad/s — adaptive threshold 용
    bool valid{false};
};

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
        int pnp_total_matches{0};   // RANSAC 입력 총 매칭 수
        double inlier_ratio{0.0};   // inliers / total (0.0~1.0)
        cv::Mat R;  // 3x3
        cv::Mat t;  // 3x1
    };

    ProcessingResult processFrame(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const CameraParams& camera_params,
                                bool first_frame = false,
                                bool enable_pose_estimation = true,
                                const ImuPredictedPose& imu_pred = {});

    void setPreviousFrame(const cv::Mat& frame, const cv::Mat& depth,
                         const Features& features);

private:
    // 컴포넌트들
    std::shared_ptr<FeatureDetector> feature_detector_;
    std::shared_ptr<FeatureMatcher> feature_matcher_;

    cv::Mat prev_frame_gray_;
    cv::Mat prev_depth_;
    Features prev_features_;
    std::vector<int> prev_track_ages_;  // Phase 5: keypoint별 연속 추적 프레임 수

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