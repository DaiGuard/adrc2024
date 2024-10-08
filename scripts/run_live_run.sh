#! /bin/env bash

if [[ $# -ne 2 ]]; then
    echo -e "Usage:"
    echo -e "  ./run_live_run.sh MODEL_PATH MASK_FILE"
    echo -e ""
    echo -e "Example:"
    echo -e "  ./run_live_run.sh models/model.pth data/mask.png"

    exit
fi

cd $(dirname $0)/../adrc_ws && \
source /opt/ros/humble/setup.bash && \
source install/local_setup.bash && \
export ROS_DOMAIN_ID=1 && \
ros2 launch adrc_ros adrc_live_run.launch.py model_path:=$1 mask_file:=$2
