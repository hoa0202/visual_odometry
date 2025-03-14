#pragma once

#include <opencv2/core.hpp>
#include "visual_odometry/types.hpp"

namespace vo {

class Visualizer {
public:
    Visualizer();
    
    // 윈도우 설정
    void setWindowSize(int width, int height);
    void setWindowPosition(int x, int y);
    
    // 시각화 옵션 설정
    void setShowOriginal(bool show) { show_original_ = show; }
    void setShowFeatures(bool show) { show_features_ = show; }
    void setShowMatches(bool show) { show_matches_ = show; }
    
    // 시각화 수행
    void visualize(const cv::Mat& original_frame,
                  const Features& features,
                  const FeatureMatches& matches,
                  const cv::Mat& prev_frame);
                  
    // 윈도우 생성/제거
    void createWindows();
    void destroyWindows();

    void setDisplayScale(float scale) { display_scale_ = scale; }
    void setUseResize(bool use) { use_resize_ = use; }

private:
    // 윈도우 설정
    int window_width_{800};
    int window_height_{600};
    int window_pos_x_{100};
    int window_pos_y_{100};
    
    // 표시 옵션
    bool show_original_{true};
    bool show_features_{true};
    bool show_matches_{true};
    
    // 윈도우 이름
    const std::string original_window_{"Original Image"};
    const std::string features_window_{"Feature Detection"};
    const std::string matches_window_{"Feature Matches"};
    
    // 시각화 설정
    float display_scale_{0.5};  // 디스플레이 스케일 (50%)
    bool use_resize_{true};     // 리사이징 사용 여부
    
    // 내부 시각화 메서드
    void showOriginalFrame(const cv::Mat& frame);
    void showFeatures(const cv::Mat& frame, const Features& features);
    void showMatches(const cv::Mat& prev_frame, 
                    const cv::Mat& curr_frame,
                    const FeatureMatches& matches);
                    
    // 버퍼
    cv::Mat display_buffer_;
};

} // namespace vo 