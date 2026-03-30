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
    
    bool getImages(cv::Mat& rgb, cv::Mat& depth, sl::SensorsData* sensors = nullptr);
    bool getSensorsDataCurrent(sl::SensorsData& sensors);
    bool getCameraParameters(cv::Mat& K, cv::Mat& D);
    bool getResolution(int& width, int& height);
    bool getCameraImuTransform(cv::Mat& R_cam_imu, cv::Mat& t_cam_imu);

private:
    sl::Camera zed_;
    sl::RuntimeParameters runtime_params_;
    bool is_connected_{false};
    
    cv::Mat slMat2cvMat(sl::Mat& input);
};

} // namespace vo 