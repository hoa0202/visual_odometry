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
    Features result;
    
    try {
        // 이미지 전처리 최적화
        cv::Mat gray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame;
        }

        // FAST 검출기 파라미터 최적화
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(gray, keypoints, fast_threshold, true);  // true: non-max suppression 활성화

        // 최적의 특징점 선택
        if (keypoints.size() > static_cast<size_t>(max_features)) {
            // 품질 기반 정렬
            std::sort(keypoints.begin(), keypoints.end(),
                     [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                         return a.response > b.response;
                     });
            keypoints.resize(max_features);
        }

        // 특징점 계산
        cv::Mat descriptors;
        if (!keypoints.empty()) {
            descriptor_->compute(gray, keypoints, descriptors);
        }

        result.keypoints = std::move(keypoints);
        result.descriptors = std::move(descriptors);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                    "Error in detectFeatures: %s", e.what());
    }

    return result;
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