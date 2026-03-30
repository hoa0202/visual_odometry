#include "visual_odometry/frame_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

namespace vo {

// Legacy: descriptor matching 모드
FrameProcessor::FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                             std::shared_ptr<FeatureMatcher> matcher)
    : feature_detector_(detector)
    , feature_matcher_(matcher)
    , use_klt_(false) {
    if (!detector || !matcher) {
        throw std::runtime_error("Null detector or matcher");
    }
}

// KLT tracker 모드
FrameProcessor::FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                             std::shared_ptr<FeatureTracker> tracker)
    : feature_detector_(detector)
    , feature_tracker_(tracker)
    , use_klt_(true) {
    if (!detector || !tracker) {
        throw std::runtime_error("Null detector or tracker");
    }
}

FrameProcessor::ProcessingResult
FrameProcessor::processFrame(const cv::Mat& rgb,
                            const cv::Mat& depth,
                            const CameraParams& camera_params,
                            bool first_frame,
                            bool enable_pose_estimation,
                            const ImuPredictedPose& imu_pred) {
    if (use_klt_) {
        return processFrameKLT(rgb, depth, camera_params, first_frame,
                               enable_pose_estimation, imu_pred);
    }
    return processFrameDescriptor(rgb, depth, camera_params, first_frame,
                                  enable_pose_estimation, imu_pred);
}

// ─── KLT Tracker 모드 ───────────────────────────────────────────────────────

FrameProcessor::ProcessingResult
FrameProcessor::processFrameKLT(const cv::Mat& rgb,
                                const cv::Mat& depth,
                                const CameraParams& camera_params,
                                bool first_frame,
                                bool enable_pose_estimation,
                                const ImuPredictedPose& imu_pred) {
    ProcessingResult result;

    cv::Mat gray = preprocessFrame(rgb);

    if (first_frame || prev_frame_gray_.empty()) {
        feature_tracker_->initialize(gray);
        result.features = detectFeatures(gray);
        setPreviousFrame(rgb, depth, result.features);
        return result;
    }

    // KLT tracking (IMU-guided when available)
    auto track_start = std::chrono::steady_clock::now();
    FeatureTracker::TrackingResult track_result;
    if (imu_pred.valid && !imu_pred.R.empty() && camera_params.fx > 0) {
        track_result = feature_tracker_->track(
            gray, imu_pred.R, camera_params.fx, camera_params.fy,
            camera_params.cx, camera_params.cy);
    } else {
        track_result = feature_tracker_->track(gray);
    }
    result.feature_matching_time = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - track_start).count();

    result.matches = track_result.matches;
    result.track_ages = track_result.track_ages;

    // Keyframe 생성용: filtering 전 원본 보존
    result.klt_curr_points = track_result.matches.curr_points;
    result.klt_track_ids = track_result.track_ids;

    auto logger = rclcpp::get_logger("frame_processor");
    static auto klt_clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    RCLCPP_INFO_THROTTLE(logger, *klt_clock, 3000,
        "KLT: tracked=%d, new=%d, fb_rejected=%d, matches=%zu",
        track_result.total_tracked, track_result.new_detected,
        track_result.fb_rejected, result.matches.prev_points.size());

    // PnP pose estimation
    if (enable_pose_estimation && !result.matches.empty() &&
        camera_params.fx > 0 && camera_params.fy > 0) {

        // === ORB-SLAM3 2-Stage Tracking ===
        //
        // Stage 1: TrackWithMotionModel — frame-to-frame depth PnP → rough pose
        //   KLT already tracked features. Backproject curr depth → 3D.
        //   solvePnP → T_prev_from_curr → T_rough = T_prev * T_rel (accurate absolute pose)
        //
        // Stage 2: TrackLocalMap — project map points with rough pose → tight matching
        //   T_rough로 map point project (3px radius) → 수백 개 매칭
        //   solvePnP with map points → final accurate absolute pose

        bool map_pnp_done = false;
        cv::Mat T_rough;  // Stage 1 결과: rough absolute pose

        // --- Stage 1: TrackWithMotionModel (frame-to-frame depth PnP) ---
        if (!depth.empty() && !result.matches.empty()) {
            backprojectAndFilter(result.matches, depth, camera_params, /*use_curr=*/true);

            // PnP: frame-to-frame
            if (!result.matches.prev_points_3d.empty() &&
                result.matches.prev_points_3d.size() >= 4) {
                cv::Mat K = camera_params.getCameraMatrix();
                cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
                cv::Mat rvec, tvec;
                std::vector<int> inliers;

                // 1차: RANSAC
                result.pnp_success = cv::solvePnPRansac(
                    result.matches.prev_points_3d,
                    result.matches.prev_points,
                    K, distCoeffs, rvec, tvec,
                    false, 200, 6.0, 0.99, inliers);

                // 2차: inlier만으로 iterative refinement
                if (result.pnp_success && static_cast<int>(inliers.size()) >= 8) {
                    std::vector<cv::Point3f> inlier_3d;
                    std::vector<cv::Point2f> inlier_2d;
                    for (int idx : inliers) {
                        inlier_3d.push_back(result.matches.prev_points_3d[idx]);
                        inlier_2d.push_back(result.matches.prev_points[idx]);
                    }
                    cv::solvePnP(inlier_3d, inlier_2d, K, distCoeffs,
                                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
                }

                result.pnp_inliers = static_cast<int>(inliers.size());
                result.pnp_total_matches = static_cast<int>(result.matches.prev_points_3d.size());
                result.inlier_ratio = (result.pnp_total_matches > 0)
                    ? static_cast<double>(result.pnp_inliers) / result.pnp_total_matches
                    : 0.0;
                if (result.pnp_success && !rvec.empty() && !tvec.empty()) {
                    cv::Rodrigues(rvec, result.R);
                    result.t = tvec;
                    result.use_map_pnp = false;

                    // Stage 1 결과 → rough absolute pose (T_world_from_camera)
                    // T_rel = T_prev_from_curr → T_rough = T_prev * T_rel
                    if (has_prev_pose_ && !prev_rvec_cw_.empty()) {
                        cv::Mat R_cw_prev;
                        cv::Rodrigues(prev_rvec_cw_, R_cw_prev);
                        cv::Mat R_wc_prev = R_cw_prev.t();
                        cv::Mat t_wc_prev = -R_wc_prev * prev_tvec_cw_;
                        cv::Mat T_prev = cv::Mat::eye(4, 4, CV_64F);
                        R_wc_prev.copyTo(T_prev(cv::Rect(0,0,3,3)));
                        t_wc_prev.copyTo(T_prev(cv::Rect(3,0,1,3)));

                        cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
                        result.R.copyTo(T_rel(cv::Rect(0,0,3,3)));
                        result.t.copyTo(T_rel(cv::Rect(3,0,1,3)));

                        T_rough = T_prev * T_rel;
                    }
                }
            }
        }

        // F2M fallback: if Stage1 failed but we have a previous pose, use it for Stage2
        if (T_rough.empty() && has_prev_pose_ && !prev_rvec_cw_.empty()) {
            cv::Mat R_cw_prev;
            cv::Rodrigues(prev_rvec_cw_, R_cw_prev);
            cv::Mat R_wc_prev = R_cw_prev.t();
            cv::Mat t_wc_prev = -R_wc_prev * prev_tvec_cw_;
            T_rough = cv::Mat::eye(4, 4, CV_64F);
            R_wc_prev.copyTo(T_rough(cv::Rect(0,0,3,3)));
            t_wc_prev.copyTo(T_rough(cv::Rect(3,0,1,3)));

            if (imu_pred.valid && !imu_pred.R.empty()) {
                cv::Mat T_delta = cv::Mat::eye(4, 4, CV_64F);
                imu_pred.R.copyTo(T_delta(cv::Rect(0,0,3,3)));
                if (!imu_pred.t.empty())
                    imu_pred.t.copyTo(T_delta(cv::Rect(3,0,1,3)));
                T_rough = T_rough * T_delta;
            }
        }

        // --- Stage 2: TrackLocalMap (map point projection with rough pose) ---
        if (!T_rough.empty() && local_map_ && local_map_->numMapPoints() > 0) {
            std::vector<cv::Point3f> map_3d;
            std::vector<cv::Point2f> map_2d;
            int n_map = local_map_->getCorrespondencesByProjection(
                T_rough, track_result.matches.curr_points,
                camera_params.fx, camera_params.fy,
                camera_params.cx, camera_params.cy,
                15.0,  // Stage 1 rough pose 오차 감안 (KLT는 descriptor 없이 proximity 매칭)
                map_3d, map_2d);

            if (n_map >= 10) {
                cv::Mat K = camera_params.getCameraMatrix();
                cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
                // T_rough → T_cw for solvePnP initial guess
                cv::Mat R_wc = T_rough(cv::Rect(0,0,3,3));
                cv::Mat t_wc = (cv::Mat_<double>(3,1) << T_rough.at<double>(0,3),
                    T_rough.at<double>(1,3), T_rough.at<double>(2,3));
                cv::Mat R_cw = R_wc.t();
                cv::Mat t_cw = -R_cw * t_wc;
                cv::Mat rvec_guess, tvec_guess = t_cw;
                cv::Rodrigues(R_cw, rvec_guess);

                std::vector<int> inliers2;
                bool ok2 = cv::solvePnPRansac(
                    map_3d, map_2d, K, distCoeffs, rvec_guess, tvec_guess,
                    true, 200, 8.0, 0.99, inliers2);

                if (ok2 && static_cast<int>(inliers2.size()) >= 8) {
                    // Iterative refinement
                    std::vector<cv::Point3f> in_3d;
                    std::vector<cv::Point2f> in_2d;
                    for (int idx : inliers2) {
                        in_3d.push_back(map_3d[idx]);
                        in_2d.push_back(map_2d[idx]);
                    }
                    cv::solvePnP(in_3d, in_2d, K, distCoeffs,
                                 rvec_guess, tvec_guess, true, cv::SOLVEPNP_ITERATIVE);

                    // Stage 2 성공 → Map PnP 결과로 override
                    result.pnp_success = true;
                    result.pnp_inliers = static_cast<int>(in_3d.size());
                    result.pnp_total_matches = n_map;
                    result.inlier_ratio = static_cast<double>(result.pnp_inliers) / n_map;
                    cv::Rodrigues(rvec_guess, result.R);
                    result.t = tvec_guess;
                    result.use_map_pnp = true;
                    result.map_correspondences = n_map;
                    map_pnp_done = true;

                    static auto s2_clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
                    RCLCPP_INFO_THROTTLE(logger, *s2_clock, 3000,
                        "Stage2 Map PnP: %d/%d inliers (%.0f%%), map=%zu",
                        result.pnp_inliers, n_map,
                        result.inlier_ratio * 100.0, local_map_->numMapPoints());
                }
            }
        }
    }

    result.features = detectFeatures(gray);
    setPreviousFrame(rgb, depth, result.features);
    return result;
}

// ─── Descriptor Matching 모드 (Legacy) ──────────────────────────────────────

FrameProcessor::ProcessingResult
FrameProcessor::processFrameDescriptor(const cv::Mat& rgb,
                                       const cv::Mat& depth,
                                       const CameraParams& camera_params,
                                       bool first_frame,
                                       bool enable_pose_estimation,
                                       const ImuPredictedPose& imu_pred) {
    ProcessingResult result;

    cv::Mat gray = preprocessFrame(rgb);

    {
        auto detect_start = std::chrono::steady_clock::now();
        result.features = detectFeatures(gray);
        result.feature_detection_time = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - detect_start).count();
    }

    if (!first_frame && !prev_frame_gray_.empty()) {
        auto match_start = std::chrono::steady_clock::now();
        result.matches = matchFeatures(result.features, gray);
        result.feature_matching_time = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - match_start).count();

        // Phase 5: Multi-view consistency — track age 계산
        std::vector<int> curr_keypoint_ages(result.features.keypoints.size(), 0);
        for (const auto& m : result.matches.matches) {
            int prev_age = (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(prev_track_ages_.size()))
                           ? prev_track_ages_[m.queryIdx] : 0;
            if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(curr_keypoint_ages.size())) {
                curr_keypoint_ages[m.trainIdx] = prev_age + 1;
            }
        }

        const int min_track_length = 3;
        if (result.matches.size() > 80) {
            int mature_count = 0;
            for (size_t i = 0; i < result.matches.size(); ++i) {
                int tIdx = result.matches.matches[i].trainIdx;
                if (tIdx >= 0 && tIdx < static_cast<int>(curr_keypoint_ages.size()) &&
                    curr_keypoint_ages[tIdx] >= min_track_length) {
                    mature_count++;
                }
            }

            if (mature_count >= 50) {
                FeatureMatches filtered;
                int short_removed = 0;
                for (size_t i = 0; i < result.matches.size(); ++i) {
                    int tIdx = result.matches.matches[i].trainIdx;
                    if (tIdx >= 0 && tIdx < static_cast<int>(curr_keypoint_ages.size()) &&
                        curr_keypoint_ages[tIdx] >= min_track_length) {
                        filtered.matches.push_back(result.matches.matches[i]);
                        filtered.prev_points.push_back(result.matches.prev_points[i]);
                        filtered.curr_points.push_back(result.matches.curr_points[i]);
                    } else {
                        short_removed++;
                    }
                }
                if (short_removed > 0) {
                    result.matches = filtered;
                    auto logger = rclcpp::get_logger("frame_processor");
                    static auto mv_clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
                    RCLCPP_INFO_THROTTLE(logger, *mv_clock, 2000,
                        "multi-view: %d mature / %zu total, removed %d short-track (age<%d)",
                        mature_count, result.matches.size() + short_removed,
                        short_removed, min_track_length);
                }
            }
        }

        prev_track_ages_ = std::move(curr_keypoint_ages);

        if (enable_pose_estimation && !result.matches.empty() && !depth.empty() &&
            camera_params.fx > 0 && camera_params.fy > 0) {
            backprojectAndFilter(result.matches, depth, camera_params, /*use_curr=*/true);

            // IMU-guided feature filtering
            if (imu_pred.valid && !imu_pred.R.empty() && !imu_pred.t.empty() &&
                result.matches.prev_points_3d.size() >= 100) {
                cv::Mat K = camera_params.getCameraMatrix();
                cv::Mat rvec_imu;
                cv::Rodrigues(imu_pred.R, rvec_imu);
                std::vector<cv::Point2f> projected;
                cv::projectPoints(result.matches.prev_points_3d, rvec_imu, imu_pred.t,
                                  K, cv::noArray(), projected);

                const double base_threshold = 10.0;
                const double reproj_threshold = base_threshold +
                    imu_pred.angular_rate * 5.0;

                std::vector<std::pair<double, size_t>> errors;
                errors.reserve(projected.size());
                for (size_t k = 0; k < projected.size(); ++k) {
                    double err = cv::norm(projected[k] - result.matches.prev_points[k]);
                    errors.emplace_back(err, k);
                }

                size_t min_retain = std::max(static_cast<size_t>(100),
                    static_cast<size_t>(projected.size() * 0.6));
                min_retain = std::min(min_retain, projected.size());

                std::vector<size_t> keep_indices;
                for (const auto& [err, idx] : errors) {
                    if (err <= reproj_threshold) {
                        keep_indices.push_back(idx);
                    }
                }

                if (keep_indices.size() < min_retain) {
                    std::sort(errors.begin(), errors.end());
                    keep_indices.clear();
                    for (size_t i = 0; i < min_retain && i < errors.size(); ++i) {
                        keep_indices.push_back(errors[i].second);
                    }
                    std::sort(keep_indices.begin(), keep_indices.end());
                }

                int removed = static_cast<int>(projected.size() - keep_indices.size());
                if (removed > 0) {
                    std::vector<cv::Point3f> filtered_3d;
                    std::vector<cv::Point2f> filtered_prev;
                    std::vector<cv::Point2f> filtered_curr;
                    std::vector<cv::DMatch> filtered_matches;
                    for (size_t idx : keep_indices) {
                        filtered_3d.push_back(result.matches.prev_points_3d[idx]);
                        filtered_prev.push_back(result.matches.prev_points[idx]);
                        if (idx < result.matches.curr_points.size())
                            filtered_curr.push_back(result.matches.curr_points[idx]);
                        if (idx < result.matches.matches.size())
                            filtered_matches.push_back(result.matches.matches[idx]);
                    }
                    result.matches.prev_points_3d = filtered_3d;
                    result.matches.prev_points = filtered_prev;
                    result.matches.curr_points = filtered_curr;
                    result.matches.matches = filtered_matches;
                    auto logger = rclcpp::get_logger("frame_processor");
                    static auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
                    RCLCPP_INFO_THROTTLE(logger, *clock, 2000,
                        "IMU-guided: removed %d/%zu features (%.1f%%), thresh=%.1fpx, remaining=%zu",
                        removed, projected.size(),
                        100.0 * removed / projected.size(),
                        reproj_threshold, keep_indices.size());
                }
            }

            // PnP: solvePnPRansac
            if (!result.matches.prev_points_3d.empty() &&
                result.matches.prev_points_3d.size() >= 4) {
                cv::Mat K = camera_params.getCameraMatrix();
                cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
                cv::Mat rvec, tvec;
                std::vector<int> inliers;
                result.pnp_success = cv::solvePnPRansac(
                    result.matches.prev_points_3d,
                    result.matches.prev_points,
                    K, distCoeffs, rvec, tvec,
                    false, 100, 8.0, 0.99, inliers);
                result.pnp_inliers = static_cast<int>(inliers.size());
                result.pnp_total_matches = static_cast<int>(result.matches.prev_points_3d.size());
                result.inlier_ratio = (result.pnp_total_matches > 0)
                    ? static_cast<double>(result.pnp_inliers) / result.pnp_total_matches
                    : 0.0;
                if (result.pnp_success && !rvec.empty() && !tvec.empty()) {
                    cv::Rodrigues(rvec, result.R);
                    result.t = tvec;
                }
            }
        }
    }

    setPreviousFrame(rgb, depth, result.features);
    return result;
}

// ─── 공통 유틸리티 ──────────────────────────────────────────────────────────

cv::Mat FrameProcessor::preprocessFrame(const cv::Mat& rgb) {
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

Features FrameProcessor::detectFeatures(const cv::Mat& gray) {
    return feature_detector_->detectFeatures(gray);
}

FeatureMatches FrameProcessor::matchFeatures(const Features& curr_features,
                                           const cv::Mat& curr_gray) {
    return feature_matcher_->match(prev_features_, curr_features,
                                 prev_frame_gray_, curr_gray);
}

void FrameProcessor::setPreviousFrame(const cv::Mat& frame,
                                     const cv::Mat& depth,
                                     const Features& features) {
    cv::cvtColor(frame, prev_frame_gray_, cv::COLOR_BGR2GRAY);
    if (!depth.empty()) {
        prev_depth_ = depth.clone();
    }
    prev_features_ = features;
}

void FrameProcessor::backprojectAndFilter(FeatureMatches& matches,
                                          const cv::Mat& depth,
                                          const CameraParams& camera_params,
                                          bool use_curr_points) {
    const double fx = camera_params.fx;
    const double fy = camera_params.fy;
    const double cx = camera_params.cx;
    const double cy = camera_params.cy;

    const float min_depth = 50.0f;
    const float max_depth = 20000.0f;

    std::vector<cv::DMatch> valid_matches;
    std::vector<cv::Point2f> valid_prev, valid_curr;
    std::vector<cv::Point3f> valid_3d;

    const auto& sample_points = use_curr_points ? matches.curr_points : matches.prev_points;

    for (size_t i = 0; i < sample_points.size(); ++i) {
        const auto& pt = sample_points[i];
        int u = static_cast<int>(std::round(pt.x));
        int v = static_cast<int>(std::round(pt.y));

        if (u < 0 || u >= depth.cols || v < 0 || v >= depth.rows) {
            continue;
        }

        float z = depth.at<float>(v, u);
        if (std::isnan(z) || std::isinf(z) || z <= 0.0f || z < min_depth || z > max_depth) {
            continue;
        }

        float x = static_cast<float>((pt.x - cx) * z / fx);
        float y = static_cast<float>((pt.y - cy) * z / fy);

        if (i < matches.matches.size())
            valid_matches.push_back(matches.matches[i]);
        valid_prev.push_back(matches.prev_points[i]);
        valid_curr.push_back(matches.curr_points[i]);
        valid_3d.emplace_back(x, y, z);
    }

    if (valid_3d.empty()) {
        matches.prev_points_3d.clear();
        return;
    }

    matches.matches = std::move(valid_matches);
    matches.prev_points = std::move(valid_prev);
    matches.curr_points = std::move(valid_curr);
    matches.prev_points_3d = std::move(valid_3d);
}

} // namespace vo
