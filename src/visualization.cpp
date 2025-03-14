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

void Visualizer::visualize(const cv::Mat& rgb,
                          const Features& features,
                          const FeatureMatches& matches,
                          const cv::Mat& prev_frame) {
    try {
        cv::Mat display_image;
        
        // 리사이징 적용
        if (use_resize_) {
            cv::resize(rgb, display_image, cv::Size(), display_scale_, display_scale_);
        } else {
            display_image = rgb.clone();
        }

        // 원본 이미지 표시
        if (show_original_) {
            cv::imshow(original_window_, display_image);
        }

        // 특징점 시각화
        if (show_features_) {
            cv::Mat feature_image = display_image.clone();
            showFeatures(feature_image, features);
            cv::imshow(features_window_, feature_image);
        }

        // 매칭 결과 시각화
        if (show_matches_ && !prev_frame.empty()) {
            cv::Mat prev_resized;
            if (use_resize_) {
                cv::resize(prev_frame, prev_resized, cv::Size(), display_scale_, display_scale_);
            } else {
                prev_resized = prev_frame.clone();
            }
            showMatches(prev_resized, display_image, matches);
        }

        cv::waitKey(1);
    } catch (const std::exception& e) {
        throw std::runtime_error("Visualization error: " + std::string(e.what()));
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

void Visualizer::showFeatures(const cv::Mat& frame, const Features& features) {
    if (frame.empty()) return;
    
    try {
        // 특징점 그리기
        for (const auto& kp : features.keypoints) {
            cv::Point2f pt(kp.pt.x * display_scale_,
                          kp.pt.y * display_scale_);
            cv::circle(frame, pt, 3, cv::Scalar(0, 0, 255), -1);
        }
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
    
    try {
        // 두 이미지를 가로로 연결
        cv::Mat combined_img;
        cv::hconcat(prev_frame, curr_frame, combined_img);

        // 매칭 선과 특징점 그리기
        for (size_t i = 0; i < matches.matches.size(); i++) {
            if (i >= matches.prev_points.size() || i >= matches.curr_points.size()) {
                continue;
            }

            // 이전 프레임의 특징점 좌표
            cv::Point2f pt1(matches.prev_points[i].x * display_scale_,
                           matches.prev_points[i].y * display_scale_);
            
            // 현재 프레임의 특징점 좌표
            cv::Point2f pt2(matches.curr_points[i].x * display_scale_ + prev_frame.cols,
                           matches.curr_points[i].y * display_scale_);
            
            // 매칭 선 그리기
            cv::line(combined_img, pt1, pt2, cv::Scalar(0, 255, 0), 1);
            cv::circle(combined_img, pt1, 3, cv::Scalar(255, 0, 0), -1);
            cv::circle(combined_img, pt2, 3, cv::Scalar(255, 0, 0), -1);
        }
        
        cv::imshow(matches_window_, combined_img);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"), 
                    "Error in showMatches: %s", e.what());
    }
}

} // namespace vo 