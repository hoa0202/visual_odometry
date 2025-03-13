#include "visual_odometry/image_processor.hpp"
#include <opencv2/imgproc.hpp>

namespace vo {

ProcessedImages ImageProcessor::processImage(const cv::Mat& input, bool enhance) {
    ProcessedImages result;
    
    // 그레이스케일 변환
    convertToGray(input);
    result.gray = gray_buffer_;

    if (enhance) {
        // 이미지 향상
        enhanceImage();
        result.enhanced = enhanced_buffer_;

        // 노이즈 제거
        denoiseImage();
        result.denoised = denoised_buffer_;

        // 마스크 생성
        createMask();
        result.masked = masked_buffer_;
    }

    return result;
}

void ImageProcessor::convertToGray(const cv::Mat& input) {
    if (input.channels() == 3) {
        if (gray_buffer_.empty() || gray_buffer_.size() != input.size()) {
            gray_buffer_.create(input.size(), CV_8UC1);
        }
        cv::cvtColor(input, gray_buffer_, cv::COLOR_BGR2GRAY);
    } else {
        input.copyTo(gray_buffer_);
    }
}

void ImageProcessor::enhanceImage() {
    if (enable_histogram_eq_) {
        if (enhanced_buffer_.empty() || enhanced_buffer_.size() != gray_buffer_.size()) {
            enhanced_buffer_.create(gray_buffer_.size(), CV_8UC1);
        }
        cv::equalizeHist(gray_buffer_, enhanced_buffer_);
    } else {
        gray_buffer_.copyTo(enhanced_buffer_);
    }
}

void ImageProcessor::denoiseImage() {
    if (gaussian_blur_size_ > 1) {
        if (denoised_buffer_.empty() || denoised_buffer_.size() != enhanced_buffer_.size()) {
            denoised_buffer_.create(enhanced_buffer_.size(), CV_8UC1);
        }
        cv::GaussianBlur(enhanced_buffer_, denoised_buffer_,
                        cv::Size(gaussian_blur_size_, gaussian_blur_size_),
                        gaussian_sigma_);
    } else {
        enhanced_buffer_.copyTo(denoised_buffer_);
    }
}

void ImageProcessor::createMask() {
    if (masked_buffer_.empty() || masked_buffer_.size() != denoised_buffer_.size()) {
        masked_buffer_.create(denoised_buffer_.size(), CV_8UC1);
    }
    // ROI 마스크 생성
    masked_buffer_ = cv::Mat::zeros(denoised_buffer_.size(), CV_8UC1);
    cv::rectangle(masked_buffer_,
                 cv::Point(0, denoised_buffer_.rows/4),
                 cv::Point(denoised_buffer_.cols, denoised_buffer_.rows*3/4),
                 cv::Scalar(255),
                 -1);
    
    // 마스크 적용
    cv::bitwise_and(denoised_buffer_, masked_buffer_, masked_buffer_);
}

} // namespace vo 