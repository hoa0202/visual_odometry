#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vo {
struct ProcessedImages {
    cv::Mat gray;
    cv::Mat enhanced;
    cv::Mat denoised;
    cv::Mat masked;
};

struct CameraParams {
    cv::Mat K;
    cv::Mat D;
    double fx, fy, cx, cy;
};

struct Features {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat visualization;  // 시각화를 위한 이미지
};

// 매칭 결과를 저장하는 구조체 추가
struct FeatureMatches {
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;
    cv::Mat visualization;  // 매칭 시각화용 (선택적)
};
} // namespace vo 