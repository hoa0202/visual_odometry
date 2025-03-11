#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

FeatureDetector::FeatureDetector() {
    // 초기 detector 생성
    updateDetector();
}

Features FeatureDetector::detectFeatures(const cv::Mat& image, 
                                       int max_features,
                                       int fast_threshold) {
    Features features;
    
    // FAST 검출기 파라미터 설정
    detector_->setMaxFeatures(max_features);
    detector_->setFastThreshold(fast_threshold);
    
    // 특징점 검출
    detector_->detect(image, features.keypoints);
    
    // 특징점 계산
    descriptor_->compute(image, features.keypoints, features.descriptors);
    
    // 시각화
    features.visualization = cv::Mat::zeros(image.size(), CV_8UC3);
    if (image.channels() == 1) {
        cv::cvtColor(image, features.visualization, cv::COLOR_GRAY2BGR);
    } else {
        image.copyTo(features.visualization);
    }
    
    // 특징점 그리기
    if (visualization_flags_ == static_cast<int>(cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS)) {
        cv::drawKeypoints(features.visualization, features.keypoints, 
                         features.visualization,
                         cv::Scalar(0, 255, 0),
                         cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    } else {
        for (const auto& kp : features.keypoints) {
            cv::circle(features.visualization, kp.pt, 3, cv::Scalar(0,255,0), -1);
        }
    }
    
    return features;
}

void FeatureDetector::setMaxFeatures(int max_features) {
    detector_ = cv::ORB::create(
        max_features,     // nfeatures
        1.2f,            // scaleFactor
        8,               // nlevels
        31,             // edgeThreshold
        0,              // firstLevel
        2,              // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,             // patchSize
        20              // fastThreshold
    );
}

void FeatureDetector::setScaleFactor(float scale_factor) {
    scale_factor_ = scale_factor;
    updateDetector();
}

void FeatureDetector::setNLevels(int n_levels) {
    n_levels_ = n_levels;
    updateDetector();
}

void FeatureDetector::setVisualizationType(const std::string& type) {
    visualization_flags_ = static_cast<int>(
        type == "rich" ? 
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS : 
        cv::DrawMatchesFlags::DEFAULT);
}

void FeatureDetector::updateDetector() {
    detector_ = cv::ORB::create(
        max_features_,     // 최대 특징점 수
        scale_factor_,     // 스케일 팩터
        n_levels_          // 피라미드 레벨 수
    );
    
    descriptor_ = detector_;  // 동일한 ORB 객체 사용
}

FeatureMatches FeatureDetector::matchFeatures(const Features& prev_features,
                                            const Features& curr_features,
                                            float ratio_threshold,
                                            bool crossCheck) {
    FeatureMatches matches;
    
    if (prev_features.descriptors.empty() || curr_features.descriptors.empty()) {
        return matches;
    }
    
    try {
        // matcher를 매번 새로 생성하지 않고 재사용
        if (!matcher_) {
            matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
        }

        if (crossCheck) {
            // 교차 검사를 사용하는 경우 직접 매칭
            std::vector<cv::DMatch> direct_matches;
            matcher_->match(prev_features.descriptors, curr_features.descriptors, 
                          direct_matches);
            
            // 교차 검사 수행
            std::vector<cv::DMatch> reverse_matches;
            matcher_->match(curr_features.descriptors, prev_features.descriptors, 
                          reverse_matches);
            
            // 양방향 매칭이 일치하는 것만 선택
            for (const auto& forward_match : direct_matches) {
                const auto& reverse_match = reverse_matches[forward_match.trainIdx];
                if (reverse_match.trainIdx == forward_match.queryIdx) {
                    matches.matches.push_back(forward_match);
                }
            }
        } else {
            // k-최근접 이웃 매칭
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatch(prev_features.descriptors, curr_features.descriptors, 
                             knn_matches, 2);
            
            // 비율 테스트 적용
            matches.matches.reserve(knn_matches.size());
            for (const auto& knn_match : knn_matches) {
                if (knn_match.size() >= 2 && 
                    knn_match[0].distance < ratio_threshold * knn_match[1].distance) {
                    matches.matches.push_back(knn_match[0]);
                }
            }
        }
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                    "OpenCV error in matchFeatures: %s", e.what());
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                    "Error in matchFeatures: %s", e.what());
    }
    
    return matches;
}

} // namespace vo 