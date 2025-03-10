#include "visual_odometry/image_processor.hpp"
#include <opencv2/imgproc.hpp>

namespace vo {

ImageProcessor::ImageProcessor() {}

ProcessedImages ImageProcessor::process(const cv::Mat& input_frame) {
    cv::Mat gray_buffer;
    return process(input_frame, gray_buffer);
}

ProcessedImages ImageProcessor::process(const cv::Mat& input_frame, cv::Mat& gray_buffer) {
    ProcessedImages result;
    
    // 그레이스케일 변환 (버퍼 재사용)
    if (gray_buffer.empty() || gray_buffer.size() != input_frame.size()) {
        gray_buffer.create(input_frame.size(), CV_8UC1);
    }
    cv::cvtColor(input_frame, gray_buffer, cv::COLOR_BGR2GRAY);
    result.gray = gray_buffer;

    // 히스토그램 평활화 (필요한 경우)
    if (enable_histogram_eq_) {
        cv::equalizeHist(result.gray, result.gray);
    }

    // 가우시안 블러 (필요한 경우)
    if (gaussian_blur_size_ > 0) {
        cv::GaussianBlur(result.gray, result.gray, 
                        cv::Size(gaussian_blur_size_, gaussian_blur_size_),
                        gaussian_sigma_);
    }

    return result;
}

cv::Mat ImageProcessor::convertToGray(const cv::Mat& input) {
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat ImageProcessor::enhanceImage(const cv::Mat& gray) {
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    return enhanced;
}

cv::Mat ImageProcessor::denoiseImage(const cv::Mat& enhanced) {
    cv::Mat denoised;
    cv::GaussianBlur(enhanced, denoised, cv::Size(5, 5), 1.0);
    return denoised;
}

cv::Mat ImageProcessor::createMask(const cv::Size& size) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    cv::rectangle(mask, 
                 cv::Point(0, size.height/4), 
                 cv::Point(size.width, size.height*3/4), 
                 cv::Scalar(255), 
                 -1);
    return mask;
}

} // namespace vo 