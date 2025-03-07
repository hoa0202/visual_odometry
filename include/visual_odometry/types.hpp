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
} // namespace vo 