import os

from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    model_path = LaunchConfiguration(
        "model_path", default="models/model.pth"
    )
    mask_file = LaunchConfiguration(
        "mask_file", default="data/mask.png"
    )

    model_path_arg = DeclareLaunchArgument("model_path", default_value="models/model.pth")
    mask_file_arg = DeclareLaunchArgument("mask_file", default_value="data/mask.png")

    live_run = Node(
        package='adrc_ros',
        executable='live_run',
        name='live_run',
        output='screen',
        emulate_tty=True,
        remappings=[
            ('/status', '/status_esp32'),
            ('/cur_vel', '/cur_vel'),
            ('/pose', '/pose_esp32'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        parameters=[{
            'model_path': model_path,     
            'mask_file': mask_file      
        }]
    )        

    return LaunchDescription([
        model_path_arg,
        mask_file_arg,

        live_run,
    ])