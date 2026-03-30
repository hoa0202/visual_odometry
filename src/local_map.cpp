#include "visual_odometry/local_map.hpp"
#include <rclcpp/rclcpp.hpp>

namespace vo {

void LocalMap::addKeyframe(int frame_id, const cv::Mat& T_world_cam,
                           const std::vector<cv::Point3f>& points_3d_cam,
                           const std::vector<int>& track_ids) {
    if (T_world_cam.empty() || points_3d_cam.empty()) return;

    cv::Mat R = T_world_cam(cv::Rect(0, 0, 3, 3));
    cv::Mat t_vec(3, 1, CV_64F);
    t_vec.at<double>(0) = T_world_cam.at<double>(0, 3);
    t_vec.at<double>(1) = T_world_cam.at<double>(1, 3);
    t_vec.at<double>(2) = T_world_cam.at<double>(2, 3);

    int added = 0, updated = 0;
    for (size_t i = 0; i < points_3d_cam.size() && i < track_ids.size(); ++i) {
        cv::Mat p_cam = (cv::Mat_<double>(3, 1) <<
            static_cast<double>(points_3d_cam[i].x),
            static_cast<double>(points_3d_cam[i].y),
            static_cast<double>(points_3d_cam[i].z));
        cv::Mat p_world = R * p_cam + t_vec;

        auto it = map_points_.find(track_ids[i]);
        if (it != map_points_.end()) {
            // 기존 map point: observation 증가, position은 유지 (첫 관측이 depth 가장 정확)
            it->second.num_observations++;
            it->second.last_frame_id = frame_id;
            updated++;
        } else {
            MapPoint mp;
            mp.world_pos = cv::Point3f(
                static_cast<float>(p_world.at<double>(0)),
                static_cast<float>(p_world.at<double>(1)),
                static_cast<float>(p_world.at<double>(2)));
            mp.track_id = track_ids[i];
            mp.num_observations = 1;
            mp.last_frame_id = frame_id;
            map_points_[track_ids[i]] = mp;
            added++;
        }
    }

    keyframe_frame_ids_.push_back(frame_id);

    // Keyframe window 관리: 오래된 keyframe 제거
    if (keyframe_frame_ids_.size() > max_keyframes_) {
        keyframe_frame_ids_.pop_front();
    }

    auto logger = rclcpp::get_logger("local_map");
    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(logger, *clock, 3000,
        "Keyframe %d: added=%d updated=%d total_map=%zu keyframes=%zu",
        frame_id, added, updated, map_points_.size(), keyframe_frame_ids_.size());
}

int LocalMap::getCorrespondences(const std::vector<cv::Point2f>& curr_points,
                                const std::vector<int>& track_ids,
                                std::vector<cv::Point3f>& map_points_3d,
                                std::vector<cv::Point2f>& image_points_2d) const {
    map_points_3d.clear();
    image_points_2d.clear();

    for (size_t i = 0; i < track_ids.size() && i < curr_points.size(); ++i) {
        auto it = map_points_.find(track_ids[i]);
        if (it != map_points_.end()) {
            map_points_3d.push_back(it->second.world_pos);
            image_points_2d.push_back(curr_points[i]);
        }
    }

    return static_cast<int>(map_points_3d.size());
}

int LocalMap::getCorrespondencesByProjection(
    const cv::Mat& T_world_cam,
    const std::vector<cv::Point2f>& curr_points,
    double fx, double fy, double cx, double cy,
    double search_radius,
    std::vector<cv::Point3f>& map_points_3d,
    std::vector<cv::Point2f>& image_points_2d) const {
    map_points_3d.clear();
    image_points_2d.clear();
    if (T_world_cam.empty() || curr_points.empty() || map_points_.empty()) return 0;

    // === ORB-SLAM3 GetFeaturesInArea: 64x48 grid 기반 spatial index ===
    const int GRID_COLS = 64;
    const int GRID_ROWS = 48;
    const float img_w = 1280.0f, img_h = 720.0f;
    const float cell_w = img_w / GRID_COLS;
    const float cell_h = img_h / GRID_ROWS;

    // Grid에 현재 feature 분배
    std::vector<std::vector<int>> grid(GRID_COLS * GRID_ROWS);
    for (int i = 0; i < static_cast<int>(curr_points.size()); ++i) {
        int gx = static_cast<int>(curr_points[i].x / cell_w);
        int gy = static_cast<int>(curr_points[i].y / cell_h);
        if (gx >= 0 && gx < GRID_COLS && gy >= 0 && gy < GRID_ROWS) {
            grid[gy * GRID_COLS + gx].push_back(i);
        }
    }

    // T_world_cam → T_cam_world
    cv::Mat R_wc = T_world_cam(cv::Rect(0, 0, 3, 3));
    cv::Mat R_cw = R_wc.t();
    double t_cw_data[3];
    {
        double twx = T_world_cam.at<double>(0, 3);
        double twy = T_world_cam.at<double>(1, 3);
        double twz = T_world_cam.at<double>(2, 3);
        t_cw_data[0] = -(R_cw.at<double>(0,0)*twx + R_cw.at<double>(0,1)*twy + R_cw.at<double>(0,2)*twz);
        t_cw_data[1] = -(R_cw.at<double>(1,0)*twx + R_cw.at<double>(1,1)*twy + R_cw.at<double>(1,2)*twz);
        t_cw_data[2] = -(R_cw.at<double>(2,0)*twx + R_cw.at<double>(2,1)*twy + R_cw.at<double>(2,2)*twz);
    }
    // R_cw 캐시
    double r00 = R_cw.at<double>(0,0), r01 = R_cw.at<double>(0,1), r02 = R_cw.at<double>(0,2);
    double r10 = R_cw.at<double>(1,0), r11 = R_cw.at<double>(1,1), r12 = R_cw.at<double>(1,2);
    double r20 = R_cw.at<double>(2,0), r21 = R_cw.at<double>(2,1), r22 = R_cw.at<double>(2,2);

    std::vector<bool> used(curr_points.size(), false);
    float r2 = static_cast<float>(search_radius * search_radius);

    for (const auto& [tid, mp] : map_points_) {
        // p_cam = R_cw * p_world + t_cw
        double wpx = mp.world_pos.x, wpy = mp.world_pos.y, wpz = mp.world_pos.z;
        double pz = r20*wpx + r21*wpy + r22*wpz + t_cw_data[2];
        if (pz < 50.0 || pz > 20000.0) continue;

        double px = r00*wpx + r01*wpy + r02*wpz + t_cw_data[0];
        double py = r10*wpx + r11*wpy + r12*wpz + t_cw_data[1];

        float u = static_cast<float>(fx * px / pz + cx);
        float v = static_cast<float>(fy * py / pz + cy);
        if (u < 0 || u >= img_w || v < 0 || v >= img_h) continue;

        // Grid cell 범위 계산 (search_radius 반영)
        int min_gx = std::max(0, static_cast<int>((u - search_radius) / cell_w));
        int max_gx = std::min(GRID_COLS - 1, static_cast<int>((u + search_radius) / cell_w));
        int min_gy = std::max(0, static_cast<int>((v - search_radius) / cell_h));
        int max_gy = std::min(GRID_ROWS - 1, static_cast<int>((v + search_radius) / cell_h));

        // Grid cell 내에서 nearest feature 탐색
        int best_idx = -1;
        float best_d2 = r2;
        for (int gy = min_gy; gy <= max_gy; ++gy) {
            for (int gx = min_gx; gx <= max_gx; ++gx) {
                for (int idx : grid[gy * GRID_COLS + gx]) {
                    if (used[idx]) continue;
                    float dx = curr_points[idx].x - u;
                    float dy = curr_points[idx].y - v;
                    float d2 = dx*dx + dy*dy;
                    if (d2 < best_d2) { best_d2 = d2; best_idx = idx; }
                }
            }
        }

        if (best_idx >= 0) {
            map_points_3d.push_back(mp.world_pos);
            image_points_2d.push_back(curr_points[best_idx]);
            used[best_idx] = true;
        }
    }

    return static_cast<int>(map_points_3d.size());
}

bool LocalMap::shouldBeKeyframe(int num_map_correspondences, int total_tracked,
                                int frames_since_last_kf) const {
    if (frames_since_last_kf < min_keyframe_interval_) return false;
    if (keyframe_frame_ids_.empty()) return true;

    // ORB-SLAM3 NeedNewKeyFrame: 충분한 map correspondence가 있으면 새 keyframe 불필요
    // 정지 시 overlap이 높으므로 keyframe 안 만듦 → drift 방지
    double overlap_ratio = (total_tracked > 0)
        ? static_cast<double>(num_map_correspondences) / total_tracked
        : 0.0;

    // 비율 70% 이상이면 현재 map으로 충분 → keyframe 불필요
    if (overlap_ratio > 0.7 && num_map_correspondences >= 80) return false;

    // 절대 수 30 미만 또는 비율 40% 미만 → 새 keyframe 필요
    return num_map_correspondences < 30 || overlap_ratio < 0.4;
}

void LocalMap::cullOldPoints(int current_frame_id, int max_unseen_frames) {
    int culled = 0;
    for (auto it = map_points_.begin(); it != map_points_.end();) {
        if (current_frame_id - it->second.last_frame_id > max_unseen_frames) {
            it = map_points_.erase(it);
            culled++;
        } else {
            ++it;
        }
    }

    if (culled > 0) {
        auto logger = rclcpp::get_logger("local_map");
        static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(logger, *clock, 5000,
            "Culled %d old map points, remaining=%zu", culled, map_points_.size());
    }
}

void LocalMap::updateObservations(int frame_id, const std::vector<int>& visible_track_ids) {
    for (int tid : visible_track_ids) {
        auto it = map_points_.find(tid);
        if (it != map_points_.end()) {
            it->second.last_frame_id = frame_id;
        }
    }
}

void LocalMap::reset() {
    map_points_.clear();
    keyframe_frame_ids_.clear();
}

}  // namespace vo
