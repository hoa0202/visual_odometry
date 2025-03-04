#include <rclcpp/rclcpp.hpp>
#include "visual_odometry/vo_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<vo::VisualOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 