#include "visual_odometry/image_processor.hpp"
#include <opencv2/imgproc.hpp>

namespace vo {

ImageProcessor::ImageProcessor() {}

ProcessedImages ImageProcessor::process(const cv::Mat& input_frame) {
    ProcessedImages result;
    
    result.gray = convertToGray(input_frame);
    result.enhanced = enhanceImage(result.gray);
    result.denoised = denoiseImage(result.enhanced);
    result.masked = result.denoised.clone();
    cv::Mat mask = createMask(result.denoised.size());
    result.denoised.copyTo(result.masked, mask);
    
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