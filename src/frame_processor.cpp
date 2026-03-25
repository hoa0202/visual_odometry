#include "visual_odometry/frame_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FrameProcessor::FrameProcessor(std::shared_ptr<FeatureDetector> detector,
                             std::shared_ptr<FeatureMatcher> matcher)
    : feature_detector_(detector)
    , feature_matcher_(matcher) {
    if (!detector || !matcher) {
        throw std::runtime_error("Null detector or matcher");
    }
}

FrameProcessor::ProcessingResult
FrameProcessor::processFrame(const cv::Mat& rgb,
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
        // 각 keypoint의 연속 추적 프레임 수를 계산하고, 짧은 track 필터링
        std::vector<int> curr_keypoint_ages(result.features.keypoints.size(), 0);
        for (const auto& m : result.matches.matches) {
            int prev_age = (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(prev_track_ages_.size()))
                           ? prev_track_ages_[m.queryIdx] : 0;
            if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(curr_keypoint_ages.size())) {
                curr_keypoint_ages[m.trainIdx] = prev_age + 1;
            }
        }

        // Multi-view filter: age >= min_track_length만 유지
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

        // track ages 저장 (다음 프레임용, filtering 전 전체 ages 사용)
        prev_track_ages_ = std::move(curr_keypoint_ages);

        // 3D 점 생성 + PnP (enable_pose_estimation 시에만)
        if (enable_pose_estimation && !result.matches.empty() && !depth.empty() &&
            camera_params.fx > 0 && camera_params.fy > 0) {
            backprojectAndFilter(result.matches, depth, camera_params, /*use_curr=*/true);

            // Phase 4: IMU-guided feature filtering (IMU gyro + prev VO translation)
            // PnP 전에 IMU-predicted pose로 각 feature를 검증, 동적 물체 feature 제거
            // 최소 100개 매칭이 있을 때만 적용 (적으면 필터링이 해로움)
            if (imu_pred.valid && !imu_pred.R.empty() && !imu_pred.t.empty() &&
                result.matches.prev_points_3d.size() >= 100) {
                cv::Mat K = camera_params.getCameraMatrix();
                cv::Mat rvec_imu;
                cv::Rodrigues(imu_pred.R, rvec_imu);
                std::vector<cv::Point2f> projected;
                cv::projectPoints(result.matches.prev_points_3d, rvec_imu, imu_pred.t,
                                  K, cv::noArray(), projected);

                // Adaptive threshold: base 10px + 회전속도 비례
                const double base_threshold = 10.0;
                const double reproj_threshold = base_threshold +
                    imu_pred.angular_rate * 5.0;  // 1 rad/s → +5px

                // 각 feature의 reprojection error 계산
                std::vector<std::pair<double, size_t>> errors;
                errors.reserve(projected.size());
                for (size_t k = 0; k < projected.size(); ++k) {
                    double err = cv::norm(projected[k] - result.matches.prev_points[k]);
                    errors.emplace_back(err, k);
                }

                // 최소 보장: 전체의 60% 또는 100개 이상 유지 (보수적)
                size_t min_retain = std::max(static_cast<size_t>(100),
                    static_cast<size_t>(projected.size() * 0.6));
                min_retain = std::min(min_retain, projected.size());

                // threshold로 1차 필터링
                std::vector<size_t> keep_indices;
                for (const auto& [err, idx] : errors) {
                    if (err <= reproj_threshold) {
                        keep_indices.push_back(idx);
                    }
                }

                // 최소 보장 미달 시 error 낮은 순으로 보충
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

    // ZED SDK: mm 단위 (7948=7.9m). min 50mm, max 20m
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

        valid_matches.push_back(matches.matches[i]);
        valid_prev.push_back(matches.prev_points[i]);
        valid_curr.push_back(matches.curr_points[i]);
        valid_3d.emplace_back(x, y, z);
    }

    if (valid_3d.empty()) {
        matches.prev_points_3d.clear();
        static bool depth_debug_logged = false;
        if (!depth_debug_logged && !sample_points.empty()) {
            float z0 = 0, z1 = 0, z2 = 0;
            int u0 = static_cast<int>(std::round(sample_points[0].x));
            int v0 = static_cast<int>(std::round(sample_points[0].y));
            if (u0 >= 0 && u0 < depth.cols && v0 >= 0 && v0 < depth.rows) {
                z0 = depth.at<float>(v0, u0);
            }
            if (sample_points.size() > 1) {
                int u1 = static_cast<int>(std::round(sample_points[1].x));
                int v1 = static_cast<int>(std::round(sample_points[1].y));
                if (u1 >= 0 && u1 < depth.cols && v1 >= 0 && v1 < depth.rows) {
                    z1 = depth.at<float>(v1, u1);
                }
            }
            if (sample_points.size() > 2) {
                int u2 = static_cast<int>(std::round(sample_points[2].x));
                int v2 = static_cast<int>(std::round(sample_points[2].y));
                if (u2 >= 0 && u2 < depth.cols && v2 >= 0 && v2 < depth.rows) {
                    z2 = depth.at<float>(v2, u2);
                }
            }
            RCLCPP_WARN(rclcpp::get_logger("frame_processor"),
                "backproject: 0 valid 3D (depth sample: %.3f, %.3f, %.3f at %zu matches, depth %dx%d)",
                z0, z1, z2, sample_points.size(), depth.cols, depth.rows);
            depth_debug_logged = true;
        }
        return;
    }

    matches.matches = std::move(valid_matches);
    matches.prev_points = std::move(valid_prev);
    matches.curr_points = std::move(valid_curr);
    matches.prev_points_3d = std::move(valid_3d);
}

} // namespace vo 