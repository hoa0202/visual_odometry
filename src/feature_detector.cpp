#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FeatureDetector::FeatureDetector() {
    try {
        // FastFeatureDetector 생성
        detector_ = cv::FastFeatureDetector::create(20);
        if (!detector_) {
            throw std::runtime_error("Failed to create FastFeatureDetector");
        }

        // ORB 생성
        descriptor_ = cv::ORB::create(
            max_features_,    // nfeatures
            scale_factor_,    // scaleFactor
            n_levels_,        // nlevels
            31,              // edgeThreshold
            0,               // firstLevel
            2,               // WTA_K
            cv::ORB::HARRIS_SCORE,  // scoreType
            31,              // patchSize
            20               // fastThreshold
        );
        if (!descriptor_) {
            throw std::runtime_error("Failed to create ORB detector");
        }

        // BFMatcher 생성
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        if (!matcher_) {
            throw std::runtime_error("Failed to create BFMatcher");
        }

        RCLCPP_INFO(rclcpp::get_logger("feature_detector"), 
                    "Feature detector initialized successfully");
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                     "Error initializing feature detector: %s", e.what());
        throw;
    }
}

Features FeatureDetector::detectFeatures(const cv::Mat& frame, 
                                       int max_features,
                                       int fast_threshold) {
    Features current_features;
    
    // 현재 프레임 특징점 검출
    keypoints_buffer_.clear();
    detector_->setThreshold(fast_threshold);
    detector_->detect(frame, keypoints_buffer_);
    
    // 최대 특징점 수 제한
    if (keypoints_buffer_.size() > static_cast<size_t>(max_features)) {
        std::sort(keypoints_buffer_.begin(), keypoints_buffer_.end(),
                 [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                     return a.response > b.response;
                 });
        keypoints_buffer_.resize(max_features);
    }
    
    // 디스크립터 계산
    descriptor_->compute(frame, keypoints_buffer_, descriptors_buffer_);
    
    // 현재 특징점 저장
    current_features.keypoints = keypoints_buffer_;
    current_features.descriptors = descriptors_buffer_;

    // 첫 프레임이면 이전 프레임으로 저장하고 리턴
    if (first_frame_) {
        prev_features_ = current_features;
        first_frame_ = false;
        return current_features;
    }

    // 이전 프레임과 매칭
    auto matches = matchFeatures(prev_features_, current_features);

    // 현재 프레임을 이전 프레임으로 저장
    prev_features_ = current_features;
    
    return current_features;
}

FeatureMatches FeatureDetector::matchFeatures(const Features& prev_features,
                                            const Features& curr_features) {
    FeatureMatches result;
    
    if (prev_features.keypoints.empty() || curr_features.keypoints.empty()) {
        return result;
    }
    
    // 매칭 수행
    std::vector<cv::DMatch> matches;
    matcher_->match(prev_features.descriptors, curr_features.descriptors, matches);
    
    // 매칭점 좌표 저장
    result.matches = matches;
    result.prev_points.reserve(matches.size());
    result.curr_points.reserve(matches.size());
    
    for (const auto& match : matches) {
        result.prev_points.push_back(prev_features.keypoints[match.queryIdx].pt);
        result.curr_points.push_back(curr_features.keypoints[match.trainIdx].pt);
    }
    
    return result;
}

void FeatureDetector::setVisualizationType(const std::string& type) {
    visualization_flags_ = static_cast<int>(
        type == "rich" ? 
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS : 
        cv::DrawMatchesFlags::DEFAULT);
}

void FeatureDetector::updateDetector() {
    try {
        // FastFeatureDetector 업데이트
        detector_ = cv::FastFeatureDetector::create(20);
        if (!detector_) {
            throw std::runtime_error("Failed to update FastFeatureDetector");
        }

        // ORB 업데이트
        descriptor_ = cv::ORB::create(
            max_features_,    // nfeatures
            scale_factor_,    // scaleFactor
            n_levels_,        // nlevels
            31,              // edgeThreshold
            0,               // firstLevel
            2,               // WTA_K
            cv::ORB::HARRIS_SCORE,  // scoreType
            31,              // patchSize
            20               // fastThreshold
        );
        if (!descriptor_) {
            throw std::runtime_error("Failed to update ORB detector");
        }

        // BFMatcher 업데이트
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        if (!matcher_) {
            throw std::runtime_error("Failed to update BFMatcher");
        }

        RCLCPP_INFO(rclcpp::get_logger("feature_detector"), 
                    "Feature detector updated successfully");
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                     "Error updating feature detector: %s", e.what());
        throw;
    }
}

} // namespace vo 