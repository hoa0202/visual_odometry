#pragma once

#include <opencv2/core.hpp>
#include <unordered_map>
#include <deque>
#include <vector>

namespace vo {

/** 3D map point (world frame, optical coordinate, mm 단위). */
struct MapPoint {
    cv::Point3f world_pos;      // 3D world frame (optical, mm)
    int track_id;               // KLT track ID
    int num_observations{1};    // keyframe observation count
    int last_frame_id{0};       // last seen frame
};

/** Local Map: keyframe 기반 3D map point 관리.
 *  KLT track_id로 2D-3D correspondence를 즉시 조회.
 *  map PnP: solvePnP(world_3d, curr_2d) → T_camera_from_world (absolute pose). */
class LocalMap {
public:
    LocalMap() = default;

    /** Keyframe의 3D points를 world frame으로 변환하여 map에 추가.
     *  @param frame_id  현재 프레임 번호
     *  @param T_world_cam  4x4, T_0_from_curr (optical, mm) — world←camera transform
     *  @param points_3d_cam  3D camera frame (optical, mm)
     *  @param track_ids  각 point의 KLT track ID */
    void addKeyframe(int frame_id, const cv::Mat& T_world_cam,
                     const std::vector<cv::Point3f>& points_3d_cam,
                     const std::vector<int>& track_ids);

    /** track_id 기반 correspondence 조회 (legacy). */
    int getCorrespondences(const std::vector<cv::Point2f>& curr_points,
                          const std::vector<int>& track_ids,
                          std::vector<cv::Point3f>& map_points_3d,
                          std::vector<cv::Point2f>& image_points_2d) const;

    /** ORB-SLAM3 style: projection 기반 매칭.
     *  모든 map point를 현재 pose로 project → 가장 가까운 KLT feature와 매칭.
     *  track_id 연속성에 의존하지 않으므로 이동 시에도 높은 매칭률 유지.
     *  @param T_world_cam  4x4, T_world_from_camera (optical, mm)
     *  @param curr_points  현재 프레임의 2D KLT tracked points
     *  @param camera_params  fx, fy, cx, cy
     *  @param search_radius  탐색 반경 (pixels). ORB-SLAM3 RGB-D: 3-15
     *  @param map_points_3d  출력: 매칭된 world 3D points
     *  @param image_points_2d  출력: 매칭된 2D observations
     *  @return 매칭 수 */
    int getCorrespondencesByProjection(
        const cv::Mat& T_world_cam,
        const std::vector<cv::Point2f>& curr_points,
        double fx, double fy, double cx, double cy,
        double search_radius,
        std::vector<cv::Point3f>& map_points_3d,
        std::vector<cv::Point2f>& image_points_2d) const;

    /** Keyframe 생성 기준 판단.
     *  map correspondence 비율이 낮거나 절대 수가 부족하면 true. */
    bool shouldBeKeyframe(int num_map_correspondences, int total_tracked,
                          int frames_since_last_kf) const;

    /** 오래 안 보인 map point 제거 */
    void cullOldPoints(int current_frame_id, int max_unseen_frames = 60);

    /** 현재 프레임에서 보인 map point의 last_frame_id 갱신 */
    void updateObservations(int frame_id, const std::vector<int>& visible_track_ids);

    size_t numKeyframes() const { return keyframe_frame_ids_.size(); }
    size_t numMapPoints() const { return map_points_.size(); }
    bool hasTrackId(int tid) const { return map_points_.count(tid) > 0; }

    void reset();

private:
    std::unordered_map<int, MapPoint> map_points_;  // track_id → MapPoint
    std::deque<int> keyframe_frame_ids_;
    size_t max_keyframes_{10};
    int min_keyframe_interval_{3};   // 최소 keyframe 간격 (프레임 수)
};

}  // namespace vo
