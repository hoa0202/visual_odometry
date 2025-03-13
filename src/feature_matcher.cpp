#include "visual_odometry/feature_matcher.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/calib3d.hpp>

namespace vo {

FeatureMatcher::FeatureMatcher() 
    : matcher_(cv::BFMatcher::create(cv::NORM_HAMMING, false)),
      ratio_threshold_(0.7f) {  // ratio threshold 초기화 추가
    if (!matcher_) {
        throw std::runtime_error("Failed to create BFMatcher");
    }
}

FeatureMatches FeatureMatcher::match(
    const Features& prev_features,
    const Features& curr_features,
    const cv::Mat& prev_frame_gray,
    const cv::Mat& curr_frame_gray) {
    
    if (prev_features.keypoints.empty() || curr_features.keypoints.empty()) {
        return FeatureMatches();
    }

    try {
        // 1. 디스크립터 매칭
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_features.descriptors, 
                          curr_features.descriptors, 
                          knn_matches, 
                          2);  // k=2로 설정

        // 2. Lowe's ratio test로 좋은 매칭만 필터링
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::Point2f> prev_pts, curr_pts;
        
        for (const auto& knn_match : knn_matches) {
            if (knn_match.size() < 2) continue;
            
            const cv::DMatch& m = knn_match[0];
            const cv::DMatch& n = knn_match[1];
            
            // ratio test
            if (m.distance < ratio_threshold_ * n.distance) {
                good_matches.push_back(m);
                prev_pts.push_back(prev_features.keypoints[m.queryIdx].pt);
                curr_pts.push_back(curr_features.keypoints[m.trainIdx].pt);
            }
        }

        // 3. RANSAC으로 이상치 제거
        if (prev_pts.size() >= 8) {  // findFundamentalMat는 최소 8점 필요
            std::vector<uchar> inlier_mask;
            cv::findFundamentalMat(prev_pts, curr_pts, 
                                 cv::FM_RANSAC,        // RANSAC 방법 사용
                                 3.0,                  // 거리 임계값 (픽셀)
                                 0.99,                 // 신뢰도
                                 inlier_mask);         // 인라이어 마스크

            // RANSAC 결과로 필터링된 매칭만 저장
            std::vector<cv::DMatch> ransac_matches;
            std::vector<cv::Point2f> ransac_prev_pts, ransac_curr_pts;
            
            for (size_t i = 0; i < inlier_mask.size(); i++) {
                if (inlier_mask[i]) {
                    ransac_matches.push_back(good_matches[i]);
                    ransac_prev_pts.push_back(prev_pts[i]);
                    ransac_curr_pts.push_back(curr_pts[i]);
                }
            }

            // 4. 이동 거리 기반 필터링 (급격한 움직임 제거)
            std::vector<cv::DMatch> final_matches;
            std::vector<cv::Point2f> final_prev_pts, final_curr_pts;
            
            const float max_movement = 100.0f;  // 최대 허용 이동 거리 (픽셀)
            
            for (size_t i = 0; i < ransac_matches.size(); i++) {
                float dx = ransac_curr_pts[i].x - ransac_prev_pts[i].x;
                float dy = ransac_curr_pts[i].y - ransac_prev_pts[i].y;
                float distance = std::sqrt(dx*dx + dy*dy);
                
                if (distance < max_movement) {
                    final_matches.push_back(ransac_matches[i]);
                    final_prev_pts.push_back(ransac_prev_pts[i]);
                    final_curr_pts.push_back(ransac_curr_pts[i]);
                }
            }

            // 결과 반환
            FeatureMatches result;
            result.matches = final_matches;
            result.prev_points = final_prev_pts;
            result.curr_points = final_curr_pts;
            return result;
        }
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_matcher"), 
                    "OpenCV error in matching: %s", e.what());
    }

    return FeatureMatches();  // 매칭 실패시 빈 결과 반환
}

} // namespace vo