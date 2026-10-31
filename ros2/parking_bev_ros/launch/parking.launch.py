from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=PathJoinSubstitution([FindPackageShare("parking_bev_ros"), "config", "parking.yaml"])),
        DeclareLaunchArgument("profile_config", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="parking_bev_ros", executable="parking_node", name="parking_assistance", output="screen",
             parameters=[LaunchConfiguration("config"), {
                 "profile_config": ParameterValue(LaunchConfiguration("profile_config"), value_type=str),
                 "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
             }]),
    ])
