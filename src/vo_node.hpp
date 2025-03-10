class VisualOdometryNode : public rclcpp::Node {
private:
    // FPS 측정 관련 파라미터 추가
    this->declare_parameter("fps_window_size", 30);
    int fps_window_size_;  // 설정 가능한 윈도우 크기
} 