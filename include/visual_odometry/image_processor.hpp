#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor() = default;

    // 이미지 처리 메서드
    ProcessedImages process(const cv::Mat& input_frame);
    ProcessedImages process(const cv::Mat& input_frame, cv::Mat& gray_buffer);
    
private:
    // 내부 처리 메서드들
    cv::Mat convertToGray(const cv::Mat& input);
    cv::Mat enhanceImage(const cv::Mat& gray);
    cv::Mat denoiseImage(const cv::Mat& enhanced);
    cv::Mat createMask(const cv::Size& size);

    // 처리 옵션
    bool enable_histogram_eq_{true};
    int gaussian_blur_size_{5};
    double gaussian_sigma_{1.0};
};

} // namespace vo 