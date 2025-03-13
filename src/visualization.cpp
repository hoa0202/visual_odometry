#include "visual_odometry/visualization.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vo {

Visualizer::Visualizer() {}

void Visualizer::setWindowSize(int width, int height) {
    window_width_ = width;
    window_height_ = height;
}

void Visualizer::setWindowPosition(int x, int y) {
    window_pos_x_ = x;
    window_pos_y_ = y;
}

void Visualizer::createWindows() {
    if (show_original_) {
        cv::namedWindow(original_window_, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(original_window_, window_pos_x_, window_pos_y_);
    }
    
    if (show_features_) {
        cv::namedWindow(features_window_, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(features_window_, 
                      window_pos_x_ + window_width_ + 10, 
                      window_pos_y_);
    }
    
    if (show_matches_) {
        cv::namedWindow(matches_window_, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(matches_window_,
                      window_pos_x_ + 2 * (window_width_ + 10),
                      window_pos_y_);
    }
}

void Visualizer::destroyWindows() {
    if (show_original_) cv::destroyWindow(original_window_);
    if (show_features_) cv::destroyWindow(features_window_);
    if (show_matches_) cv::destroyWindow(matches_window_);
}

void Visualizer::visualize(const cv::Mat& original_frame,
                         const Features& features,
                         const FeatureMatches& matches,
                         const cv::Mat& prev_frame) {
    // 입력 이미지 유효성 검사 추가
    if (original_frame.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("visualizer"), 
                    "Empty original frame received");
        return;
    }

    if (window_width_ <= 0 || window_height_ <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "Invalid window size: %dx%d", window_width_, window_height_);
        return;
    }

    try {
        if (show_original_) {
            showOriginalFrame(original_frame);
        }
        
        if (show_features_) {
            showFeatures(original_frame, features);
        }
        
        if (show_matches_ && !prev_frame.empty()) {
            showMatches(prev_frame, original_frame, matches);
        }
        
        cv::waitKey(1);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "OpenCV error in visualize: %s", e.what());
    }
}

void Visualizer::showOriginalFrame(const cv::Mat& frame) {
    if (frame.empty()) return;
    
    try {
        cv::resize(frame, display_buffer_, 
                  cv::Size(window_width_, window_height_));
        cv::imshow(original_window_, display_buffer_);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "Error in showOriginalFrame: %s", e.what());
    }
}

void Visualizer::showFeatures(const cv::Mat& frame, 
                            const Features& features) {
    if (frame.empty()) return;
    
    try {
        // 프레임 복사 및 리사이즈
        cv::resize(frame, display_buffer_,
                  cv::Size(window_width_, window_height_));
                
        // 특징점 그리기
        for (const auto& kp : features.keypoints) {
            cv::Point2f pt(kp.pt.x * window_width_ / frame.cols,
                          kp.pt.y * window_height_ / frame.rows);
            cv::circle(display_buffer_, pt, 3, cv::Scalar(0, 0, 255), -1);
        }
        
        cv::imshow(features_window_, display_buffer_);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "Error in showFeatures: %s", e.what());
    }
}

void Visualizer::showMatches(const cv::Mat& prev_frame,
                           const cv::Mat& curr_frame,
                           const FeatureMatches& matches) {
    if (prev_frame.empty() || curr_frame.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("visualizer"), "Empty frames in showMatches");
        return;
    }
    
    if (matches.matches.empty()) {
        RCLCPP_DEBUG(rclcpp::get_logger("visualizer"), "No matches to display");
        return;
    }

    try {
        // 이미지 리사이즈
        cv::Mat prev_resized, curr_resized;
        cv::resize(prev_frame, prev_resized, 
                  cv::Size(window_width_, window_height_));
        cv::resize(curr_frame, curr_resized, 
                  cv::Size(window_width_, window_height_));
        
        // 두 이미지를 가로로 연결하기 전에 색상 확인 및 변환
        if (prev_resized.channels() == 1) {
            cv::cvtColor(prev_resized, prev_resized, cv::COLOR_GRAY2BGR);
        }
        if (curr_resized.channels() == 1) {
            cv::cvtColor(curr_resized, curr_resized, cv::COLOR_GRAY2BGR);
        }
        
        // 두 이미지 가로로 연결
        cv::Mat combined_img;
        cv::hconcat(prev_resized, curr_resized, combined_img);

        if (combined_img.empty()) {
            RCLCPP_ERROR(rclcpp::get_logger("visualizer"), "Failed to create combined image");
            return;
        }
        
        // 매칭 선과 특징점 그리기
        for (size_t i = 0; i < matches.matches.size(); i++) {
            if (i >= matches.prev_points.size() || i >= matches.curr_points.size()) {
                RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                            "Invalid match index: %zu", i);
                continue;
            }

            // 이전 프레임의 특징점 좌표
            cv::Point2f pt1 = matches.prev_points[i];
            pt1.x = pt1.x * window_width_ / prev_frame.cols;
            pt1.y = pt1.y * window_height_ / prev_frame.rows;
            
            // 현재 프레임의 특징점 좌표
            cv::Point2f pt2 = matches.curr_points[i];
            pt2.x = (pt2.x * window_width_ / curr_frame.cols) + window_width_;
            pt2.y = pt2.y * window_height_ / curr_frame.rows;
            
            // 좌표가 이미지 범위 내에 있는지 확인
            if (pt1.x >= 0 && pt1.x < window_width_ && pt1.y >= 0 && pt1.y < window_height_ &&
                pt2.x >= window_width_ && pt2.x < 2 * window_width_ && pt2.y >= 0 && pt2.y < window_height_) {
                // 매칭 선 그리기
                cv::line(combined_img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                
                // 특징점 그리기
                cv::circle(combined_img, pt1, 3, cv::Scalar(255, 0, 0), -1);
                cv::circle(combined_img, pt2, 3, cv::Scalar(255, 0, 0), -1);
            }
        }
        
        // 디버그 정보 추가
        cv::putText(combined_img, 
                   "Matches: " + std::to_string(matches.matches.size()),
                   cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   1.0, 
                   cv::Scalar(0, 255, 0), 
                   2);
        
        cv::imshow(matches_window_, combined_img);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "Error in showMatches: %s", e.what());
    }
}

} // namespace vo 