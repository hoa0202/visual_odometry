#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include "visual_odometry/types.hpp"
#include <vector>

namespace vo {

/** KLT optical flow 기반 feature tracker (VINS-Mono 스타일).
 *  ORB descriptor matching 대신 pixel-level 추적 → 동일 물리점의 연속 추적 보장.
 *  track age가 자연스럽게 누적 → multi-view consistency 내장. */
class FeatureTracker {
public:
    struct Params {
        int max_features{500};         // 최대 추적 feature 수
        int min_features{100};         // 이 이하로 떨어지면 새 feature 검출
        double quality_level{0.01};    // GFTT quality (낮을수록 더 많이 검출)
        double min_distance{15.0};     // feature 간 최소 거리 (px)
        cv::Size win_size{21, 21};     // KLT window size
        int max_pyramid_level{3};      // KLT pyramid level
        double fb_threshold{1.0};      // forward-backward consistency threshold (px)
        int min_track_length{3};       // PnP에 사용할 최소 track 길이
    };

    explicit FeatureTracker(const Params& params);

    /** 이전 프레임 → 현재 프레임 추적.
     *  반환: FeatureMatches (prev_points, curr_points 채워짐, matches는 비어있음).
     *  track_ages: 각 매칭점의 연속 추적 프레임 수. */
    struct TrackingResult {
        FeatureMatches matches;        // prev_points, curr_points (PnP 호환)
        std::vector<int> track_ages;   // 각 point의 연속 추적 프레임 수
        std::vector<int> track_ids;    // 각 point의 고유 track ID (local map용)
        int total_tracked{0};          // 추적 성공 수
        int new_detected{0};           // 새로 검출된 feature 수
        int fb_rejected{0};            // forward-backward check로 제거된 수
    };

    TrackingResult track(const cv::Mat& curr_gray);

    /** IMU-guided tracking: predicted rotation warps prev points for better KLT convergence.
     *  @param imu_R  3x3 relative rotation (optical frame, prev→curr) */
    TrackingResult track(const cv::Mat& curr_gray,
                         const cv::Mat& imu_R, double fx, double fy, double cx, double cy);

    /** 첫 프레임 초기화 (feature 검출만 수행) */
    void initialize(const cv::Mat& gray);

    /** 추적 상태 리셋 */
    void reset();

    /** 현재 추적 중인 feature 수 */
    size_t numTracked() const { return tracked_pts_.size(); }

    /** 파라미터 업데이트 */
    void setParams(const Params& params) { params_ = params; }

private:
    Params params_;

    // 추적 상태
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> tracked_pts_;   // 현재 추적 중인 points
    std::vector<int> track_ids_;             // 각 point의 고유 ID
    std::vector<int> track_ages_;            // 각 point의 연속 추적 프레임 수
    int next_id_{0};

    // 새 feature 검출 (기존 feature 위치를 피해서)
    void detectNewFeatures(const cv::Mat& gray, int needed);

    // 기존 feature 근처의 mask 생성
    cv::Mat createMask(const cv::Mat& gray) const;
};

}  // namespace vo
