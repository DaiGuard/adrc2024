import os

from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    path = LaunchConfiguration(
        "path", default="data/"
    )
    timespan = LaunchConfiguration(
        "timespan", default="1.0"
    )

    path_arg = DeclareLaunchArgument("path", default_value="data/")
    timespan_arg = DeclareLaunchArgument("timespan", default_value="1.0")

    endpoint = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="endpoint",
        output="screen",
        emulate_tty=True,
    )

    front_decode = Node(
        package="image_transport",
        executable="republish",
        name="front_decode",
        arguments=[
            "compressed", "raw"
        ],
        remappings=[
            ("/in/compressed", "/front_camera/compressed"),
            ("/out", "/front_camera/raw")
        ],
    )

    rear_decode = Node(
        package="image_transport",
        executable="republish",
        name="rear_decode",
        arguments=[
            "compressed", "raw"
        ],
        remappings=[
            ("/in/compressed", "/rear_camera/compressed"),
            ("/out", "/rear_camera/raw")
        ],
    )

    data_capture = Node(
        package="adrc_ros",
        executable="data_capture",
        name="data_capture",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "path": path,
            "timespan": timespan,
        }]
    )
    
    return LaunchDescription([
        path_arg,
        timespan_arg,
        
        endpoint,
        front_decode,
        rear_decode,
        data_capture,
    ])