from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=PathJoinSubstitution([FindPackageShare("parking_bev_ros"), "config", "asyncbev.yaml"])),
        DeclareLaunchArgument("worker_command", default_value=""),
        DeclareLaunchArgument("worker_endpoint", default_value=""),
        DeclareLaunchArgument("packet_directory", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="parking_bev_ros", executable="perception_node", name="parking_bev", output="screen",
             parameters=[LaunchConfiguration("config"), {
                 "worker_command": ParameterValue(LaunchConfiguration("worker_command"), value_type=str),
                 "worker_endpoint": ParameterValue(LaunchConfiguration("worker_endpoint"), value_type=str),
                 "packet_directory": ParameterValue(LaunchConfiguration("packet_directory"), value_type=str),
                 "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
             }]),
    ])
