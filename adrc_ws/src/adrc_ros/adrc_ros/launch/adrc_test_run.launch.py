import os

from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    model_path = LaunchConfiguration(
        "model_path", default="models/model.pth"
    )
    mask_file = LaunchConfiguration(
        "mask_file", default="mask.png"
    )

    model_path_arg = DeclareLaunchArgument("model_path", default_value="models/model.pth")
    mask_file_arg = DeclareLaunchArgument("mask_file", default_value="mask.png")

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

    test_run = Node(
        package="adrc_ros",
        executable="test_run",
        name="test_run",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "model_path": model_path,
            "mask_file": mask_file
        }]
    )
    
    return LaunchDescription([
        model_path_arg,
        mask_file_arg,
        
        endpoint,
        front_decode,
        rear_decode,
        test_run,
    ])