#include "visual_odometry/feature_matcher.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/calib3d.hpp>

namespace vo {

FeatureMatcher::FeatureMatcher() 
    : matcher_(cv::BFMatcher::create(cv::NORM_HAMMING, false)) {
    if (!matcher_) {
        throw std::runtime_error("Failed to create BFMatcher");
    }
}

FeatureMatches FeatureMatcher::match(
    const Features& prev_features,
    const Features& curr_features,
    const cv::Mat& prev_frame_gray,
    const cv::Mat& curr_frame_gray) {
    
    FeatureMatches result;
    
    try {
        // 1. 입력 검증
        if (prev_features.keypoints.empty() || curr_features.keypoints.empty() ||
            prev_frame_gray.empty() || curr_frame_gray.empty()) {
            return result;
        }

        // 2. 디스크립터 매칭 수행
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_features.descriptors, curr_features.descriptors, knn_matches, 2);

        // 3. 비율 테스트로 좋은 매칭 선별
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::Point2f> prev_pts, curr_pts;
        
        const float ratio_thresh = 0.75f;
        for (const auto& matches : knn_matches) {
            if (matches.size() < 2) continue;
            
            if (matches[0].distance < ratio_thresh * matches[1].distance) {
                good_matches.push_back(matches[0]);
                prev_pts.push_back(prev_features.keypoints[matches[0].queryIdx].pt);
                curr_pts.push_back(curr_features.keypoints[matches[0].trainIdx].pt);
            }
        }

        // 4. 매칭이 충분한지 확인
        if (good_matches.size() < 8) {
            return result;
        }

        // 5. RANSAC으로 이상치 제거
        cv::Mat mask;
        cv::findFundamentalMat(
            prev_pts, curr_pts,
            cv::RANSAC,
            3.0,    // 거리 임계값 (픽셀)
            0.99,   // 신뢰도
            mask
        );

        // 6. 최종 매칭 결과 저장
        for (size_t i = 0; i < static_cast<size_t>(mask.rows); i++) {
            if (mask.at<uchar>(i)) {
                result.matches.push_back(good_matches[i]);
                result.prev_points.push_back(prev_pts[i]);
                result.curr_points.push_back(curr_pts[i]);
            }
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_matcher"),
                    "Error in matching: %s", e.what());
    }
    
    return result;
}

} // namespace vo