#include "visual_odometry/frame_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>

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
                           bool first_frame) {
    ProcessingResult result;
    
    // 1. 전처리
    cv::Mat gray = preprocessFrame(rgb);
    
    // 2. 특징점 검출
    {
        auto detect_start = std::chrono::steady_clock::now();
        result.features = detectFeatures(gray);
        result.feature_detection_time = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - detect_start).count();
    }

    // 3. 특징점 매칭 (첫 프레임이 아닌 경우)
    if (!first_frame && !prev_frame_gray_.empty()) {
        auto match_start = std::chrono::steady_clock::now();
        result.matches = matchFeatures(result.features, gray);
        result.feature_matching_time = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - match_start).count();
    }

    // 4. 현재 프레임을 이전 프레임으로 저장
    setPreviousFrame(rgb, result.features);
    
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
                                    const Features& features) {
    cv::cvtColor(frame, prev_frame_gray_, cv::COLOR_BGR2GRAY);
    prev_features_ = features;
}

} // namespace vo 