#! /bin/env bash

if [ $# -gt 0 ]; then

    if [ $1 == "-h" ]; then
        echo -e "Usage:"
        echo -e "  ./robot_sync.sh OPTIONS"
        echo -e ""
        echo -e "  OPTIONS: esp32,ros,data,models,tools"
        exit
    fi

    MODE=$1
else
    MODE="esp32,ros,scripts,data,models,tools"
fi

# 環境設定
USERNAME="nvidia"
TARGETNAME="adrc.local"
# TARGETNAME="192.168.11.4"
TARGETDIR=/home/$USERNAME/Workspaces/adrc2024/

# ワークディレクトリを取得
WORKDIR=$(cd $(dirname $0)/.. && pwd)

if [[ "$MODE" == *esp32* ]]; then
    echo -e "\e[34msynchronize ESP32 directory \e[m"
    # ESP32ソフトのアップロード
    rsync -rv --exclude=".git" --exclude=".vscode" --exclude=".pio" \
    $WORKDIR/adrc_esp32 \
    $USERNAME@$TARGETNAME:$TARGETDIR
    # ESP32ソフトのマイコン書き込み
    ssh $USERNAME@$TARGETNAME "cd $TARGETDIR/adrc_esp32/ && ~/.local/bin/pio run -t upload"
fi

if [[ "$MODE" == *ros* ]]; then
    echo -e "\e[34msynchronize ROS directory \e[m"
    # ROSソフのアップロード
    rsync -rv --exclude=".git" --exclude=".vscode" --delete \
        $WORKDIR/adrc_ws/src \
        $USERNAME@$TARGETNAME:$TARGETDIR/adrc_ws/
    ssh $USERNAME@$TARGETNAME "cd $TARGETDIR/adrc_ws/ && source /opt/ros/humble/setup.bash && colcon build --symlink-install --continue-on-error"
fi

if [[ "$MODE" == *scripts* ]]; then
    echo -e "\e[34msynchronize Scripts directory \e[m"
    # スクリプトのアップロード
    rsync -rv --exclude=".git" --exclude=".vscode" --delete \
        $WORKDIR/scripts \
        $USERNAME@$TARGETNAME:$TARGETDIR
fi

if [[ "$MODE" == *tools* ]]; then
    echo -e "\e[34msynchronize Tool directory \e[m"
    # スクリプトのアップロード
    rsync -rv --exclude=".git" --exclude=".vscode" --exclude="__pycache__" --delete \
        $WORKDIR/tools \
        $USERNAME@$TARGETNAME:$TARGETDIR/adrc_ws
fi

if [[ "$MODE" == *models* ]]; then
    echo -e "\e[34msynchronize Model directory \e[m"
    # スクリプトのアップロード
    rsync -rv --exclude=".git" --exclude=".vscode" --delete \
        $WORKDIR/models \
        $USERNAME@$TARGETNAME:$TARGETDIR/adrc_ws
fi

if [[ "$MODE" == *data* ]]; then
    echo -e "\e[34msynchronize Data directory \e[m"
    # データのダウンロード
    rsync -rv --exclude=".git" --exclude=".vscode" \
        $USERNAME@$TARGETNAME:$TARGETDIR/adrc_ws/data \
        $WORKDIR/
fi