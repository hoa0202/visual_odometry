#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

class VisualOdometryNode : public rclcpp::Node
{
public:
    VisualOdometryNode() : Node("visual_odometry_node")
    {
        // RGB 이미지 구독
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/rgb/image_rect_color", 10,
            std::bind(&VisualOdometryNode::rgbCallback, this, std::placeholders::_1));

        // Depth 이미지 구독
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/depth/depth_registered", 10,
            std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));

        // Camera Info 구독
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/zed/zed_node/rgb/camera_info", 10,
            std::bind(&VisualOdometryNode::cameraInfoCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Visual Odometry Node has been started");
        RCLCPP_INFO(this->get_logger(), "Visual Odometry Node initialized");
    }

private:

    // OpenCV 행렬 초기화
    cv::Mat K_{cv::Mat::eye(3, 3, CV_64F)};
    cv::Mat D_{cv::Mat::zeros(5, 1, CV_64F)};
    cv::Mat current_frame_;
    cv::Mat previous_frame_;
    cv::Mat current_depth_;
    cv::Mat previous_depth_;
    
    double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};
    bool camera_info_received_{false};
    int frame_count_{0};



    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_info_received_) {
            try {
                // 카메라 행렬 초기화
                K_ = cv::Mat::eye(3, 3, CV_64F);
                K_.at<double>(0,0) = msg->k[0];  // fx
                K_.at<double>(1,1) = msg->k[4];  // fy
                K_.at<double>(0,2) = msg->k[2];  // cx
                K_.at<double>(1,2) = msg->k[5];  // cy
                K_.at<double>(2,2) = 1.0;

                // 왜곡 계수 초기화
                if (!msg->d.empty()) {
                    D_ = cv::Mat(msg->d.size(), 1, CV_64F, 0.0);
                    for(size_t i = 0; i < msg->d.size(); i++) {
                        D_.at<double>(i) = msg->d[i];
                    }
                }

                fx_ = msg->k[0];
                fy_ = msg->k[4];
                cx_ = msg->k[2];
                cy_ = msg->k[5];

                camera_info_received_ = true;
                RCLCPP_INFO(this->get_logger(), "Camera parameters have been initialized");
                RCLCPP_INFO(this->get_logger(), "fx: %f, fy: %f, cx: %f, cy: %f", fx_, fy_, cx_, cy_);
            }
            catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV error in camera info callback: %s", e.what());
            }
        }
    }



    
    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!camera_info_received_) {
            return;
        }

        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGRA8);
            
            cv::Mat bgr_image;
            cv::cvtColor(cv_ptr->image, bgr_image, cv::COLOR_BGRA2BGR);

            if (!current_frame_.empty()) {
                current_frame_.copyTo(previous_frame_);
            }
            
            bgr_image.copyTo(current_frame_);

            // 이미지 처리 함수 호출
            processImage();

            RCLCPP_INFO_THROTTLE(this->get_logger(), 
                                *this->get_clock(), 
                                1000,
                                "RGB image processed. Size: %dx%d", 
                                current_frame_.cols, 
                                current_frame_.rows);
        }
        catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!camera_info_received_) {
            return;
        }

        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            
            // 이전 프레임 저장
            if (!current_depth_.empty()) {
                current_depth_.copyTo(previous_depth_);
            }
            
            // 현재 프레임 저장
            cv_ptr->image.copyTo(current_depth_);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), 
                                *this->get_clock(), 
                                1000,  // 1초마다 로그 출력
                                "Depth image processed. Size: %dx%d", 
                                current_depth_.cols, 
                                current_depth_.rows);
        }
        catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error in depth callback: %s", e.what());
        }
        catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void processImage()
    {
        if (current_frame_.empty()) {
            return;
        }

        try {
            // 1. 이미지 전처리
            cv::Mat processed_frame;
            cv::cvtColor(current_frame_, processed_frame, cv::COLOR_BGR2GRAY);  // 그레이스케일 변환
            
            // 2. 이미지 개선
            cv::Mat enhanced_frame;
            cv::equalizeHist(processed_frame, enhanced_frame);  // 히스토그램 평활화
            
            // 3. 노이즈 제거
            cv::Mat denoised_frame;
            cv::GaussianBlur(enhanced_frame, denoised_frame, cv::Size(5, 5), 1.0);

            // 4. 이미지 마스킹 (선택적)
            cv::Mat mask = cv::Mat::zeros(denoised_frame.size(), CV_8UC1);
            // ROI(Region of Interest) 설정
            cv::rectangle(mask, cv::Point(0, denoised_frame.rows/4), 
                         cv::Point(denoised_frame.cols, denoised_frame.rows*3/4), 
                         cv::Scalar(255), -1);
            
            // 마스크 적용
            cv::Mat masked_frame;
            denoised_frame.copyTo(masked_frame, mask);

            RCLCPP_INFO(this->get_logger(), "Image processing completed");
        }
        catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error in image processing: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisualOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}