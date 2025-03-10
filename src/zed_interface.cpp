#include "visual_odometry/zed_interface.hpp"

namespace vo {

ZEDInterface::ZEDInterface() {}

ZEDInterface::~ZEDInterface() {
    disconnect();
}

bool ZEDInterface::connect(int serial_number, 
                         sl::RESOLUTION resolution,
                         int fps,
                         sl::DEPTH_MODE depth_mode) {
    sl::InitParameters init_params;
    init_params.camera_resolution = resolution;
    init_params.camera_fps = fps;
    init_params.depth_mode = depth_mode;
    init_params.input.setFromSerialNumber(serial_number);
    
    // 성능 최적화 설정 추가
    init_params.sdk_gpu_id = 0;
    init_params.sdk_verbose = false;
    init_params.camera_disable_self_calib = true;
    init_params.depth_stabilization = false;  // 깊이 안정화 비활성화
    init_params.enable_image_enhancement = false;  // 이미지 향상 비활성화
    init_params.grab_compute_capping_fps = 0;  // FPS 제한 해제
    
    sl::ERROR_CODE err = zed_.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        return false;
    }

    // 카메라 설정 상세 정보 출력
    auto camera_info = zed_.getCameraInformation();
    std::cout << "\nZED Camera Configuration:" << std::endl;
    std::cout << "- Requested FPS: " << fps << std::endl;
    std::cout << "- Actual FPS: " << camera_info.camera_configuration.fps << std::endl;
    std::cout << "- Resolution: " << camera_info.camera_configuration.resolution.width 
              << "x" << camera_info.camera_configuration.resolution.height << std::endl;
    std::cout << "- Camera Model: " << camera_info.camera_model << std::endl;
    std::cout << "- Serial Number: " << camera_info.serial_number << std::endl;
    
    // 현재 카메라 설정값 출력
    int current_gain, current_exposure;
    zed_.getCameraSettings(sl::VIDEO_SETTINGS::GAIN, current_gain);
    zed_.getCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, current_exposure);
    std::cout << "- Current Gain: " << current_gain << std::endl;
    std::cout << "- Current Exposure: " << current_exposure << std::endl;
    
    // 이미지 획득 성능 최적화
    zed_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, 50);
    zed_.setCameraSettings(sl::VIDEO_SETTINGS::GAIN, 50);

    is_connected_ = true;
    return true;
}

void ZEDInterface::disconnect() {
    if (is_connected_) {
        zed_.close();
        is_connected_ = false;
    }
}

bool ZEDInterface::getImages(cv::Mat& rgb, cv::Mat& depth) {
    static sl::Mat zed_rgb;  // 정적 버퍼 재사용
    static sl::Mat zed_depth;
    
    sl::RuntimeParameters params;
    params.enable_depth = true;
    params.confidence_threshold = 50;
    params.remove_saturated_areas = false;  // 불필요한 처리 비활성화
    
    if (zed_.grab(params) != sl::ERROR_CODE::SUCCESS) {
        return false;
    }
    
    // 최소한의 처리로 이미지 획득
    zed_.retrieveImage(zed_rgb, sl::VIEW::LEFT, sl::MEM::CPU, zed_rgb.getResolution());
    zed_.retrieveMeasure(zed_depth, sl::MEASURE::DEPTH, sl::MEM::CPU, zed_depth.getResolution());
    
    // 직접 메모리 접근
    rgb = cv::Mat(zed_rgb.getHeight(), zed_rgb.getWidth(),
                 CV_8UC4, zed_rgb.getPtr<sl::uchar1>(sl::MEM::CPU));
    depth = cv::Mat(zed_depth.getHeight(), zed_depth.getWidth(),
                   CV_32FC1, zed_depth.getPtr<sl::float1>(sl::MEM::CPU));
                   
    return true;
}

bool ZEDInterface::getCameraParameters(cv::Mat& K, cv::Mat& D) {
    if (!is_connected_) return false;

    auto camera_info = zed_.getCameraInformation();
    auto left_cam = camera_info.camera_configuration.calibration_parameters.left_cam;

    K = (cv::Mat_<double>(3,3) << 
        left_cam.fx, 0, left_cam.cx,
        0, left_cam.fy, left_cam.cy,
        0, 0, 1);
    
    D = cv::Mat::zeros(5, 1, CV_64F);  // ZED는 왜곡 보정이 이미 되어있음
    
    return true;
}

cv::Mat ZEDInterface::slMat2cvMat(sl::Mat& input) {
    return cv::Mat(input.getHeight(), input.getWidth(),
                  CV_8UC4, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

bool ZEDInterface::isConnected() const {
    return is_connected_;
}

} // namespace vo 