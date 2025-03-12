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

FeatureMatches FeatureMatcher::match(const Features& prev_features,
                                   const Features& curr_features,
                                   const cv::Mat& prev_frame_gray,
                                   const cv::Mat& curr_frame_gray) {
    try {
        // 1. Optical Flow 매칭 시도
        auto matches = matchOpticalFlow(prev_features, curr_features,
                                      prev_frame_gray, curr_frame_gray);
        
        // 2. 매칭이 부족하면 디스크립터 매칭으로 보완
        if (matches.matches.size() < 10) {
            auto desc_matches = matchDescriptors(prev_features, curr_features);
            matches.matches.insert(matches.matches.end(),
                                 desc_matches.matches.begin(),
                                 desc_matches.matches.end());
            matches.prev_points.insert(matches.prev_points.end(),
                                     desc_matches.prev_points.begin(),
                                     desc_matches.prev_points.end());
            matches.curr_points.insert(matches.curr_points.end(),
                                     desc_matches.curr_points.begin(),
                                     desc_matches.curr_points.end());
        }
        
        return matches;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_matcher"),
                    "Error in match: %s", e.what());
        return FeatureMatches();
    }
}

FeatureMatches FeatureMatcher::matchOpticalFlow(
    const Features& prev_features,
    const Features& curr_features,
    const cv::Mat& prev_frame_gray,
    const cv::Mat& curr_frame_gray) {
    
    FeatureMatches result;
    
    try {
        // 1. 기본 검사
        if (prev_features.keypoints.empty() || curr_features.keypoints.empty() ||
            prev_frame_gray.empty() || curr_frame_gray.empty()) {
            return result;
        }

        // 2. 특징점 좌표 추출
        prev_pts_.clear();
        curr_pts_.clear();
        
        for (const auto& kp : prev_features.keypoints) {
            prev_pts_.push_back(kp.pt);
        }
        curr_pts_.resize(prev_pts_.size());  // 미리 크기 할당

        // 3. Optical Flow 계산
        std::vector<uchar> status;
        std::vector<float> err;
        
        cv::calcOpticalFlowPyrLK(
            prev_frame_gray, 
            curr_frame_gray,
            prev_pts_,
            curr_pts_,
            status,
            err,
            cv::Size(21, 21),
            3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)
        );

        // 4. 매칭 필터링
        std::vector<cv::Point2f> good_prev_pts, good_curr_pts;
        std::vector<cv::DMatch> good_matches;
        
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                // 기본 필터링
                if (err[i] > 10.0) continue;
                
                cv::Point2f motion = curr_pts_[i] - prev_pts_[i];
                float motion_mag = cv::norm(motion);
                
                // 모션 범위 검사
                if (motion_mag < 0.5 || motion_mag > 30.0) continue;
                
                good_matches.push_back(cv::DMatch(good_prev_pts.size(), good_prev_pts.size(), err[i]));
                good_prev_pts.push_back(prev_pts_[i]);
                good_curr_pts.push_back(curr_pts_[i]);
            }
        }

        // 5. 최종 결과 저장
        if (good_prev_pts.size() >= 8) {
            cv::Mat mask;
            cv::findFundamentalMat(good_prev_pts, good_curr_pts, 
                                 cv::RANSAC, 3.0, 0.99, mask);

            for (size_t i = 0; i < mask.rows; i++) {
                if (mask.at<uchar>(i)) {
                    result.matches.push_back(good_matches[i]);
                    result.prev_points.push_back(good_prev_pts[i]);
                    result.curr_points.push_back(good_curr_pts[i]);
                }
            }
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_matcher"),
                    "Error in matchOpticalFlow: %s", e.what());
    }
    
    return result;
}

FeatureMatches FeatureMatcher::matchDescriptors(
    const Features& prev_features,
    const Features& curr_features) {
    
    FeatureMatches result;
    
    try {
        if (prev_features.descriptors.empty() || curr_features.descriptors.empty()) {
            return result;
        }

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_features.descriptors, curr_features.descriptors, knn_matches, 2);
        
        const float ratio_thresh = 0.8f;
        for (const auto& matches : knn_matches) {
            if (matches.size() < 2) continue;
            
            if (matches[0].distance < ratio_thresh * matches[1].distance) {
                result.matches.push_back(matches[0]);
                result.prev_points.push_back(prev_features.keypoints[matches[0].queryIdx].pt);
                result.curr_points.push_back(curr_features.keypoints[matches[0].trainIdx].pt);
            }
        }

        // RANSAC으로 이상치 제거
        if (result.prev_points.size() >= 8) {
            cv::Mat mask;
            cv::findFundamentalMat(result.prev_points, result.curr_points,
                                 cv::RANSAC, 3.0, 0.99, mask);

            std::vector<cv::DMatch> good_matches;
            std::vector<cv::Point2f> good_prev_pts, good_curr_pts;

            for (size_t i = 0; i < mask.rows; i++) {
                if (mask.at<uchar>(i)) {
                    good_matches.push_back(result.matches[i]);
                    good_prev_pts.push_back(result.prev_points[i]);
                    good_curr_pts.push_back(result.curr_points[i]);
                }
            }

            result.matches = good_matches;
            result.prev_points = good_prev_pts;
            result.curr_points = good_curr_pts;
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_matcher"),
                    "Error in matchDescriptors: %s", e.what());
    }
    
    return result;
}

} // namespace vo 