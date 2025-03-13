#include "visual_odometry/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>    // calcOpticalFlowPyrLK를 위해 추가
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
            n_levels_         // nlevels
        );
        if (!descriptor_) {
            throw std::runtime_error("Failed to create ORB detector");
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
        if (frame.channels() == 3) {
            // 현재 프레임을 그레이스케일로 변환하고 저장
            cv::cvtColor(frame, curr_frame_gray_, cv::COLOR_BGR2GRAY);
        } else {
            curr_frame_gray_ = frame.clone();
        }

        // 첫 프레임이면 이전 프레임으로도 저장
        if (first_frame_) {
            prev_frame_gray_ = curr_frame_gray_.clone();
            first_frame_ = false;
        }

        // FAST 검출기 파라미터 최적화
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(curr_frame_gray_, keypoints, fast_threshold, true);

        // 최적의 특징점 선택
        if (keypoints.size() > static_cast<size_t>(max_features)) {
            std::sort(keypoints.begin(), keypoints.end(),
                     [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                         return a.response > b.response;
                     });
            keypoints.resize(max_features);
        }

        // 특징점 계산
        cv::Mat descriptors;
        if (!keypoints.empty()) {
            descriptor_->compute(curr_frame_gray_, keypoints, descriptors);
        }

        result.keypoints = std::move(keypoints);
        result.descriptors = std::move(descriptors);

        // 다음 프레임을 위해 현재 프레임을 이전 프레임으로 저장
        prev_frame_gray_ = curr_frame_gray_.clone();
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("feature_detector"), 
                    "Error in detectFeatures: %s", e.what());
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
            n_levels_         // nlevels
        );
        if (!descriptor_) {
            throw std::runtime_error("Failed to update ORB detector");
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