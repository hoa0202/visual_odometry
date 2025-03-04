#pragma once

#include <opencv2/core.hpp>

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
} // namespace vo 