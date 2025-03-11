from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 경로 가져오기
    pkg_dir = get_package_share_directory('visual_odometry')
    
    # 파라미터 파일 경로
    param_file = os.path.join(pkg_dir, 'config', 'vo_params.yaml')
    
    # 노드 설정
    vo_node = Node(
        package='visual_odometry',
        executable='vo_node',
        name='visual_odometry_node',
        parameters=[param_file],  # 파라미터 파일 로드
        output='screen'
    )
    
    return LaunchDescription([vo_node]) 