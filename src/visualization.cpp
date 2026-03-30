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
        
        if (show_matches_) {
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

void Visualizer::showMatches(const cv::Mat& /* prev_frame */,
                           const cv::Mat& curr_frame,
                           const FeatureMatches& matches) {
    if (curr_frame.empty()) return;

    try {
        cv::Mat vis;
        cv::resize(curr_frame, vis, cv::Size(window_width_, window_height_));
        if (vis.channels() == 1)
            cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

        float sx = (float)window_width_ / curr_frame.cols;
        float sy = (float)window_height_ / curr_frame.rows;

        // prev_points = inlier keypoints, curr_points = all matched keypoints
        std::set<int> inlier_px;
        for (const auto& p : matches.prev_points) {
            int key = (int)(p.x * 1000) + (int)(p.y * 1000) * 10000;
            inlier_px.insert(key);
        }

        for (const auto& p : matches.curr_points) {
            cv::Point2f pt(p.x * sx, p.y * sy);
            int key = (int)(p.x * 1000) + (int)(p.y * 1000) * 10000;
            bool is_inlier = inlier_px.count(key) > 0;
            cv::Scalar color = is_inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::circle(vis, pt, 3, color, -1, cv::LINE_AA);
        }

        char buf[128];
        snprintf(buf, sizeof(buf), "Inliers: %zu / %zu",
                 matches.prev_points.size(), matches.curr_points.size());
        cv::putText(vis, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        cv::imshow(matches_window_, vis);
    }
    catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualizer"),
                    "Error in showMatches: %s", e.what());
    }
}

} // namespace vo 