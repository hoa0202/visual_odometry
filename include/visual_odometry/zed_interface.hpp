#pragma once
#include <sl/Camera.hpp>
#include <opencv2/core.hpp>

namespace vo {

class ZEDInterface {
public:
    ZEDInterface();
    ~ZEDInterface();

    bool connect(int serial_number, 
                sl::RESOLUTION resolution = sl::RESOLUTION::HD1080,
                int fps = 30,
                sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE::ULTRA);
    void disconnect();
    bool isConnected() const;
    
    bool getImages(cv::Mat& rgb, cv::Mat& depth);
    bool getCameraParameters(cv::Mat& K, cv::Mat& D);

private:
    sl::Camera zed_;
    sl::RuntimeParameters runtime_params_;
    bool is_connected_{false};
    
    cv::Mat slMat2cvMat(sl::Mat& input);
};

} // namespace vo 