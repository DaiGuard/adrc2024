#! /bin/env bash

if [ $# -ne 2 ]; then
    echo -e "Usage:"
    echo -e "  ./run_data_capture.sh PATH TIMESTAMP"
    echo -e ""
    echo -e "Example:"
    echo -e "  ./run_data_capture.sh data/test/ 0.3"

    exit
fi

cd $(dirname $0)/../adrc_ws && \
source /opt/ros/foxy/setup.bash && \
source install/local_setup.bash && \
export ROS_DOMAIN_ID=1 && \
ros2 launch adrc_ros data_capture.launch.py path:=$1 timespan:=$2