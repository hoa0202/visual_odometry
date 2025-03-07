from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('visual_odometry')
    
    # Paths
    params_path = PathJoinSubstitution(
        [pkg_share, 'config', 'vo_params.yaml'])
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),

        # Visual Odometry Node
        Node(
            package='visual_odometry',
            executable='vo_node',
            name='visual_odometry_node',
            output='screen',
            parameters=[params_path, {
                'use_sim_time': use_sim_time,
            }],
            remappings=[
                # ZED 카메라 토픽 리매핑
                ('rgb_image', '/zed/zed_node/rgb/image_rect_color'),
                ('depth_image', '/zed/zed_node/depth/depth_registered'),
                ('camera_info', '/zed/zed_node/rgb/camera_info'),
            ]
        )
    ]) 