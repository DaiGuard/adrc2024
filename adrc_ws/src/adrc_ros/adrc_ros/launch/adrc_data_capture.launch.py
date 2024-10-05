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

    path = LaunchConfiguration(
        'path', default="data/"
    )
    timespan = LaunchConfiguration(
        'timespan', default="0.3"
    )

    path_arg = DeclareLaunchArgument("path", default_value="data/")
    timespan_arg = DeclareLaunchArgument("timespan", default_value="0.3")

    data_capture = Node(
        package='adrc_ros',
        executable='data_capture',
        name='data_capture',
        output='screen',
        emulate_tty=True,
        remappings=[
            ('/status', '/status_esp32'),
            ('/cur_vel', '/cur_vel'),
            ('/pose', '/pose_esp32'),
        ],
        parameters=[{
            "path": path,
            "timespan": timespan
        }]
    )        

    return LaunchDescription([
        path_arg,
        timespan_arg,

        data_capture,
    ])