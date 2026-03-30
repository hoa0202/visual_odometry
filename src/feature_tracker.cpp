#include "visual_odometry/feature_tracker.hpp"
#include <opencv2/calib3d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <numeric>

namespace vo {

FeatureTracker::FeatureTracker(const Params& params)
    : params_(params) {}

void FeatureTracker::initialize(const cv::Mat& gray) {
    prev_gray_ = gray.clone();
    tracked_pts_.clear();
    track_ids_.clear();
    track_ages_.clear();
    detectNewFeatures(gray, params_.max_features);
}

void FeatureTracker::reset() {
    prev_gray_ = cv::Mat();
    tracked_pts_.clear();
    track_ids_.clear();
    track_ages_.clear();
}

FeatureTracker::TrackingResult FeatureTracker::track(const cv::Mat& curr_gray) {
    TrackingResult result;

    if (prev_gray_.empty() || tracked_pts_.empty()) {
        initialize(curr_gray);
        result.new_detected = static_cast<int>(tracked_pts_.size());
        return result;
    }

    // 1. Forward KLT: prev → curr
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        prev_gray_, curr_gray, tracked_pts_, curr_pts,
        status, err, params_.win_size, params_.max_pyramid_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    // 2. Backward KLT: curr → prev (forward-backward consistency check)
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> back_status;
    std::vector<float> back_err;
    cv::calcOpticalFlowPyrLK(
        curr_gray, prev_gray_, curr_pts, back_pts,
        back_status, back_err, params_.win_size, params_.max_pyramid_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    // 3. Filter: status + FB check + boundary check
    std::vector<cv::Point2f> good_prev, good_curr;
    std::vector<int> good_ids, good_ages;
    int fb_rejected = 0;

    for (size_t i = 0; i < tracked_pts_.size(); ++i) {
        if (!status[i] || !back_status[i]) continue;

        // Forward-backward consistency
        float fb_dist = cv::norm(tracked_pts_[i] - back_pts[i]);
        if (fb_dist > params_.fb_threshold) {
            fb_rejected++;
            continue;
        }

        // Boundary check
        if (curr_pts[i].x < 0 || curr_pts[i].x >= curr_gray.cols ||
            curr_pts[i].y < 0 || curr_pts[i].y >= curr_gray.rows) {
            continue;
        }

        good_prev.push_back(tracked_pts_[i]);
        good_curr.push_back(curr_pts[i]);
        good_ids.push_back(track_ids_[i]);
        good_ages.push_back(track_ages_[i] + 1);
    }

    result.fb_rejected = fb_rejected;
    result.total_tracked = static_cast<int>(good_prev.size());

    // 4. Fundamental matrix RANSAC (skip when motion is too small — degenerate)
    if (good_prev.size() >= 8) {
        std::vector<float> flows;
        flows.reserve(good_prev.size());
        for (size_t i = 0; i < good_prev.size(); ++i)
            flows.push_back(cv::norm(good_curr[i] - good_prev[i]));
        std::nth_element(flows.begin(), flows.begin() + flows.size()/2, flows.end());
        float median_flow = flows[flows.size()/2];

        if (median_flow > 2.0f) {
            std::vector<uchar> inlier_mask;
            cv::findFundamentalMat(good_prev, good_curr, cv::FM_RANSAC, 1.0, 0.99, inlier_mask);

            std::vector<cv::Point2f> inlier_prev, inlier_curr;
            std::vector<int> inlier_ids, inlier_ages;
            for (size_t i = 0; i < inlier_mask.size(); ++i) {
                if (inlier_mask[i]) {
                    inlier_prev.push_back(good_prev[i]);
                    inlier_curr.push_back(good_curr[i]);
                    inlier_ids.push_back(good_ids[i]);
                    inlier_ages.push_back(good_ages[i]);
                }
            }
            good_prev = std::move(inlier_prev);
            good_curr = std::move(inlier_curr);
            good_ids = std::move(inlier_ids);
            good_ages = std::move(inlier_ages);
        }
    }

    // 5. 결과 채우기
    result.matches.prev_points = good_prev;
    result.matches.curr_points = good_curr;
    result.track_ages = good_ages;
    result.track_ids = good_ids;

    // 6. 상태 업데이트: curr가 다음 프레임의 prev가 됨
    tracked_pts_ = good_curr;
    track_ids_ = good_ids;
    track_ages_ = good_ages;

    // 7. Feature 부족 시 새로 검출
    int needed = params_.min_features - static_cast<int>(tracked_pts_.size());
    if (needed > 0) {
        int before = static_cast<int>(tracked_pts_.size());
        detectNewFeatures(curr_gray, needed);
        result.new_detected = static_cast<int>(tracked_pts_.size()) - before;
    }

    prev_gray_ = curr_gray.clone();
    return result;
}

FeatureTracker::TrackingResult FeatureTracker::track(
    const cv::Mat& curr_gray,
    const cv::Mat& imu_R, double fx, double fy, double cx, double cy) {

    if (imu_R.empty() || prev_gray_.empty() || tracked_pts_.empty())
        return track(curr_gray);

    // Warp prev points by IMU predicted rotation (homography-like for pure rotation)
    // For each point: unproject → rotate → reproject
    std::vector<cv::Point2f> predicted_pts;
    predicted_pts.reserve(tracked_pts_.size());
    for (const auto& pt : tracked_pts_) {
        double x = (pt.x - cx) / fx;
        double y = (pt.y - cy) / fy;
        cv::Mat p3d = (cv::Mat_<double>(3,1) << x, y, 1.0);
        cv::Mat rotated = imu_R * p3d;
        double rz = rotated.at<double>(2);
        if (rz < 0.01) {
            predicted_pts.push_back(pt);
            continue;
        }
        double u = fx * rotated.at<double>(0) / rz + cx;
        double v = fy * rotated.at<double>(1) / rz + cy;
        predicted_pts.push_back(cv::Point2f(static_cast<float>(u), static_cast<float>(v)));
    }

    // KLT with predicted initial guess
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        prev_gray_, curr_gray, tracked_pts_, predicted_pts,
        status, err, params_.win_size, params_.max_pyramid_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Backward check
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> back_status;
    std::vector<float> back_err;
    cv::calcOpticalFlowPyrLK(
        curr_gray, prev_gray_, predicted_pts, back_pts,
        back_status, back_err, params_.win_size, params_.max_pyramid_level,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    TrackingResult result;
    std::vector<cv::Point2f> good_prev, good_curr;
    std::vector<int> good_ids, good_ages;
    int fb_rejected = 0;

    for (size_t i = 0; i < tracked_pts_.size(); ++i) {
        if (!status[i] || !back_status[i]) continue;
        float fb_dist = cv::norm(tracked_pts_[i] - back_pts[i]);
        if (fb_dist > params_.fb_threshold * 2.0f) { // relaxed threshold for IMU-guided
            fb_rejected++;
            continue;
        }
        if (predicted_pts[i].x < 0 || predicted_pts[i].x >= curr_gray.cols ||
            predicted_pts[i].y < 0 || predicted_pts[i].y >= curr_gray.rows)
            continue;

        good_prev.push_back(tracked_pts_[i]);
        good_curr.push_back(predicted_pts[i]);
        good_ids.push_back(track_ids_[i]);
        good_ages.push_back(track_ages_[i] + 1);
    }

    result.fb_rejected = fb_rejected;
    result.total_tracked = static_cast<int>(good_prev.size());

    if (good_prev.size() >= 8) {
        std::vector<float> flows;
        flows.reserve(good_prev.size());
        for (size_t i = 0; i < good_prev.size(); ++i)
            flows.push_back(cv::norm(good_curr[i] - good_prev[i]));
        std::nth_element(flows.begin(), flows.begin() + flows.size()/2, flows.end());
        float median_flow = flows[flows.size()/2];

        if (median_flow > 2.0f) {
        std::vector<uchar> inlier_mask;
        cv::findFundamentalMat(good_prev, good_curr, cv::FM_RANSAC, 1.0, 0.99, inlier_mask);
        std::vector<cv::Point2f> ip, ic;
        std::vector<int> ii, ia;
        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                ip.push_back(good_prev[i]);
                ic.push_back(good_curr[i]);
                ii.push_back(good_ids[i]);
                ia.push_back(good_ages[i]);
            }
        }
        good_prev = std::move(ip);
        good_curr = std::move(ic);
        good_ids = std::move(ii);
        good_ages = std::move(ia);
        } // median_flow > 2.0f
    }

    result.matches.prev_points = good_prev;
    result.matches.curr_points = good_curr;
    result.track_ages = good_ages;
    result.track_ids = good_ids;

    tracked_pts_ = good_curr;
    track_ids_ = good_ids;
    track_ages_ = good_ages;

    int needed = params_.min_features - static_cast<int>(tracked_pts_.size());
    if (needed > 0) {
        int before = static_cast<int>(tracked_pts_.size());
        detectNewFeatures(curr_gray, needed);
        result.new_detected = static_cast<int>(tracked_pts_.size()) - before;
    }

    prev_gray_ = curr_gray.clone();
    return result;
}

void FeatureTracker::detectNewFeatures(const cv::Mat& gray, int needed) {
    if (needed <= 0) return;

    cv::Mat mask = createMask(gray);
    std::vector<cv::Point2f> new_pts;
    cv::goodFeaturesToTrack(
        gray, new_pts, needed, params_.quality_level,
        params_.min_distance, mask, 3, false, 0.04);

    for (const auto& pt : new_pts) {
        tracked_pts_.push_back(pt);
        track_ids_.push_back(next_id_++);
        track_ages_.push_back(0);
    }
}

cv::Mat FeatureTracker::createMask(const cv::Mat& gray) const {
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    int radius = static_cast<int>(params_.min_distance);
    for (const auto& pt : tracked_pts_) {
        cv::circle(mask, pt, radius, 0, -1);
    }
    return mask;
}

}  // namespace vo
