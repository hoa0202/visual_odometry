#pragma once

#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class ImageProcessor {
public:
    ImageProcessor() = default;
    ~ImageProcessor() = default;

    // 이미지 전처리 메서드
    ProcessedImages processImage(const cv::Mat& input, bool enhance = false);

    // 파라미터 설정
    void setGaussianBlurSize(int size) { gaussian_blur_size_ = size; }
    void setGaussianSigma(double sigma) { gaussian_sigma_ = sigma; }
    void setEnableHistogramEq(bool enable) { enable_histogram_eq_ = enable; }

private:
    // 버퍼 재사용을 위한 멤버 변수들
    cv::Mat gray_buffer_;
    cv::Mat enhanced_buffer_;
    cv::Mat denoised_buffer_;
    cv::Mat masked_buffer_;

    // 파라미터
    int gaussian_blur_size_{3};
    double gaussian_sigma_{1.0};
    bool enable_histogram_eq_{false};

    // 내부 처리 메서드들
    void convertToGray(const cv::Mat& input);
    void enhanceImage();
    void denoiseImage();
    void createMask();
};

} // namespace vo 