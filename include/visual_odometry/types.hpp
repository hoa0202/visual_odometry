#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <map>

namespace vo {

// ─── VIO State Machine ──────────────────────────────────────────────────────

enum class VIOState {
    NOT_INITIALIZED,
    INITIALIZING,
    TRACKING,
    RECENTLY_LOST,
    LOST
};

inline const char* vioStateStr(VIOState s) {
    switch (s) {
        case VIOState::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case VIOState::INITIALIZING:    return "INITIALIZING";
        case VIOState::TRACKING:        return "TRACKING";
        case VIOState::RECENTLY_LOST:   return "RECENTLY_LOST";
        case VIOState::LOST:            return "LOST";
    }
    return "UNKNOWN";
}

// ─── IMU Preintegration Noise (ZED2i BMI088 defaults) ───────────────────────

struct ImuPreintegrationParams {
    double accel_noise_sigma{0.05};
    double gyro_noise_sigma{0.005};
    double accel_bias_rw_sigma{0.0005};
    double gyro_bias_rw_sigma{0.00005};
    double gravity{9.81};
};

// ─── VIO Configuration ─────────────────────────────────────────────────────

struct VIOConfig {
    std::string init_mode{"full"};       // "full" | "simple" | "none"
    int init_min_keyframes{10};
    double init_max_time_sec{5.0};
    bool use_imu{true};
    int graph_window_size{50};           // Working Memory max size
    double lost_timeout_sec{5.0};
    int min_tracking_features{15};

    // Keyframe policy
    int min_keyframe_interval{3};
    double keyframe_min_map_ratio{0.5};

    // Loop closure (RTAB-Map Bayesian)
    bool loop_closure_enable{true};
    int loop_min_matches{30};
    double loop_min_score{0.3};
    int loop_min_interval_keyframes{10};
    double bayesian_threshold{0.55};     // Bayesian posterior acceptance threshold
    int bayesian_temporal_consistency{3}; // consecutive wins needed

    // Memory (RTAB-Map 4-tier)
    int stm_size{10};                    // Short-Term Memory max size
    double rehearsal_similarity{0.8};    // BoW similarity threshold for merge
    double proximity_max_dist{1.0};      // meters, for proximity link detection

    // Vocabulary
    std::string vocabulary_path{""};

    ImuPreintegrationParams imu_params;
};

// ─── VIO Output ────────────────────────────────────────────────────────────

struct VIOOutput {
    cv::Mat T_world_body;       // 4x4, body frame (m)
    cv::Mat T_world_optical;    // 4x4, optical frame (mm)
    VIOState state{VIOState::NOT_INITIALIZED};
    double tracking_quality{0.0};
    int num_tracked_features{0};
    int num_map_matches{0};
    bool is_keyframe{false};
    bool loop_closure_detected{false};
    double vx{0}, vy{0}, vz{0}; // velocity body frame (m/s)
};

// ─── RTAB-Map Memory Tier ───────────────────────────────────────────────────

enum class MemoryTier { SENSORY, STM, WM, LTM };

// ─── RTAB-Map Link Types ───────────────────────────────────────────────────

enum class LinkType { NEIGHBOR, LOOP_CLOSURE, PROXIMITY };

struct Link {
    int from_id{-1};
    int to_id{-1};
    LinkType type{LinkType::NEIGHBOR};
    cv::Mat T_relative;         // 4x4 relative pose
};

// ─── RTAB-Map Signature (replaces KeyframeNode) ────────────────────────────

struct Signature {
    int id{-1};
    cv::Mat T_world_cam;        // 4x4 optimized pose (optical, mm)
    cv::Mat descriptors;        // ORB descriptors
    std::vector<cv::KeyPoint> keypoints;
    double timestamp{0.0};
    int weight{0};              // visit count (RTAB-Map: higher = more visited)
    MemoryTier tier{MemoryTier::STM};
    std::vector<Link> links;

};

// Backward compat alias
using KeyframeNode = Signature;

// ─── IMU Data ──────────────────────────────────────────────────────────────

struct ImuData {
    double ang_vel_x{0}, ang_vel_y{0}, ang_vel_z{0};  // rad/s
    double lin_acc_x{0}, lin_acc_y{0}, lin_acc_z{0};   // m/s²
    double timestamp{0};
    bool valid{false};
};
struct ProcessedImages {
    cv::Mat gray;
    cv::Mat enhanced;
    cv::Mat denoised;
    cv::Mat masked;
};

struct CameraParams {
    double fx{0.0};  // 초점 거리 x
    double fy{0.0};  // 초점 거리 y
    double cx{0.0};  // 주점 x
    double cy{0.0};  // 주점 y
    int width{0};    // 이미지 너비
    int height{0};   // 이미지 높이
    
    // 카메라 행렬 반환
    cv::Mat getCameraMatrix() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0,0) = fx;
        K.at<double>(1,1) = fy;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;
        return K;
    }
};

struct Features {
    std::vector<cv::KeyPoint> keypoints;  // 특징점 좌표
    cv::Mat descriptors;                  // 특징점 디스크립터
    cv::Mat visualization;               // 시각화용 이미지 (선택적)
};

// 매칭 결과를 저장하는 구조체 추가
struct FeatureMatches {
    std::vector<cv::DMatch> matches;           // 매칭 결과
    std::vector<cv::Point2f> prev_points;      // 이전 프레임의 매칭점 좌표 (2D)
    std::vector<cv::Point2f> curr_points;      // 현재 프레임의 매칭점 좌표 (2D)
    std::vector<cv::Point3f> prev_points_3d;  // 이전 프레임 3D 점 (카메라 좌표계, PnP용)

    void reserve(size_t size) {
        matches.reserve(size);
        prev_points.reserve(size);
        curr_points.reserve(size);
        prev_points_3d.reserve(size);
    }

    void clear() {
        matches.clear();
        prev_points.clear();
        curr_points.clear();
        prev_points_3d.clear();
    }
    
    // 매칭 개수 반환 (KLT 모드: prev_points 기준, descriptor 모드: matches 기준)
    size_t size() const {
        return matches.empty() ? prev_points.size() : matches.size();
    }

    // 매칭이 비어있는지 확인
    bool empty() const {
        return prev_points.empty() && matches.empty();
    }
};
} // namespace vo 