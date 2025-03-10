from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('visual_odometry')
    
    # Paths
    params_path = PathJoinSubstitution(
        [pkg_share, 'config', 'vo_params.yaml'])
    
    return LaunchDescription([
        # DISPLAY 환경 변수 설정
        SetEnvironmentVariable('DISPLAY', ':0'),

        # Visual Odometry Node
        Node(
            package='visual_odometry',
            executable='vo_node',
            name='visual_odometry_node',
            output='screen',
            parameters=[params_path]
        )
    ]) 