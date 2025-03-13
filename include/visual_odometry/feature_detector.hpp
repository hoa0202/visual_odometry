#pragma once

#include <opencv2/features2d.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class FeatureDetector {
public:
    FeatureDetector();
    
    Features detectFeatures(const cv::Mat& frame,
                          int max_features = 1000,
                          int fast_threshold = 20);
                          
    void setVisualizationType(const std::string& type);
    void updateDetector();

    // Setter 함수들 추가
    void setMaxFeatures(int max_features) { 
        max_features_ = max_features; 
        updateDetector(); 
    }
    
    void setScaleFactor(float scale_factor) { 
        scale_factor_ = scale_factor; 
        updateDetector(); 
    }
    
    void setNLevels(int n_levels) { 
        n_levels_ = n_levels; 
        updateDetector(); 
    }

private:
    // 특징점 검출 관련 멤버 변수만 유지
    cv::Ptr<cv::FastFeatureDetector> detector_;
    cv::Ptr<cv::ORB> descriptor_;
    cv::Mat prev_frame_gray_;
    cv::Mat curr_frame_gray_;
    bool first_frame_{true};
    int max_features_{1000};
    float scale_factor_{1.2f};
    int n_levels_{8};
    int visualization_flags_{0};
};

} // namespace vo 