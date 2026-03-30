#pragma once

#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"
#include "visual_odometry/feature_detector.hpp"
#include "visual_odometry/feature_matcher.hpp"
#include "visual_odometry/feature_tracker.hpp"
#include "visual_odometry/local_map.hpp"
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
    // Legacy: descriptor matching 모드
    FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                  std::shared_ptr<FeatureMatcher> matcher);

    // KLT tracker 모드 (VINS-style)
    FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                  std::shared_ptr<FeatureTracker> tracker);

    struct ProcessingResult {
        Features features;
        FeatureMatches matches;
        std::vector<int> track_ages;       // KLT: 각 point의 연속 추적 프레임 수
        double feature_detection_time{0.0};
        double feature_matching_time{0.0};
        double visualization_time{0.0};
        bool is_keyframe{false};
        // PnP 결과
        bool pnp_success{false};
        int pnp_inliers{0};
        int pnp_total_matches{0};   // RANSAC 입력 총 매칭 수
        double inlier_ratio{0.0};   // inliers / total (0.0~1.0)
        cv::Mat R;  // 3x3
        cv::Mat t;  // 3x1
        // Phase B: Local Map PnP
        bool use_map_pnp{false};    // true: R,t = T_camera_from_world (absolute pose)
                                     // false: R,t = T_prev_from_curr (relative pose)
        int map_correspondences{0}; // map point correspondence 수
        // KLT tracked points (keyframe 생성용, filtering 전 원본)
        std::vector<cv::Point2f> klt_curr_points;
        std::vector<int> klt_track_ids;
    };

    ProcessingResult processFrame(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const CameraParams& camera_params,
                                bool first_frame = false,
                                bool enable_pose_estimation = true,
                                const ImuPredictedPose& imu_pred = {});

    /** Local map 설정 (non-owning). nullptr이면 frame-to-frame PnP만 사용. */
    void setLocalMap(LocalMap* map) { local_map_ = map; }

    /** 이전 프레임의 최적화된 pose 설정 (Map PnP initial guess용).
     *  T_camera_from_world = [R|t] (solvePnP convention, optical frame, mm) */
    void setPreviousPose(const cv::Mat& rvec_cw, const cv::Mat& tvec_cw) {
        prev_rvec_cw_ = rvec_cw.clone();
        prev_tvec_cw_ = tvec_cw.clone();
        has_prev_pose_ = true;
    }

    void setPreviousFrame(const cv::Mat& frame, const cv::Mat& depth,
                         const Features& features);

private:
    // 컴포넌트들
    std::shared_ptr<FeatureDetector> feature_detector_;
    std::shared_ptr<FeatureMatcher> feature_matcher_;   // descriptor matching 모드
    std::shared_ptr<FeatureTracker> feature_tracker_;   // KLT tracking 모드
    bool use_klt_{false};
    LocalMap* local_map_{nullptr};

    cv::Mat prev_frame_gray_;
    cv::Mat prev_depth_;
    Features prev_features_;
    std::vector<int> prev_track_ages_;  // descriptor matching 모드용 (legacy)

    // ORB-SLAM3 style: 이전 pose를 PnP initial guess로 사용
    cv::Mat prev_rvec_cw_;  // T_camera_from_world rotation (Rodrigues)
    cv::Mat prev_tvec_cw_;  // T_camera_from_world translation
    bool has_prev_pose_{false};

    // KLT 모드 processFrame
    ProcessingResult processFrameKLT(const cv::Mat& rgb,
                                     const cv::Mat& depth,
                                     const CameraParams& camera_params,
                                     bool first_frame,
                                     bool enable_pose_estimation,
                                     const ImuPredictedPose& imu_pred);

    // Descriptor matching 모드 processFrame (legacy)
    ProcessingResult processFrameDescriptor(const cv::Mat& rgb,
                                            const cv::Mat& depth,
                                            const CameraParams& camera_params,
                                            bool first_frame,
                                            bool enable_pose_estimation,
                                            const ImuPredictedPose& imu_pred);

    void backprojectAndFilter(FeatureMatches& matches,
                              const cv::Mat& depth,
                              const CameraParams& camera_params,
                              bool use_curr_points = false);
    cv::Mat preprocessFrame(const cv::Mat& rgb);
    Features detectFeatures(const cv::Mat& gray);
    FeatureMatches matchFeatures(const Features& curr_features,
                               const cv::Mat& curr_gray);
};

} // namespace vo
