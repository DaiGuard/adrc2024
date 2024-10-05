#! /bin/env bash

gst-launch-1.0 -v -e nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=60/1' ! m.sink_0 \
    nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=60/1' ! nvvideoconvert flip-method=rotate-180 ! m.sink_1 \
    nvstreammux name=m width=1280 height=1440 batch-size=2 num-surfaces-per-frame=1 \
    ! nvmultistreamtiler columns=1 rows=2 width=1280 height=1440 \
    ! nvvideoconvert ! 'video/x-raw(memory:NVMM),width=640,height=960' ! tee name=t ! queue ! fakesink sync=false \
    t.src_1 ! queue ! nvvidconv ! 'video/x-raw,width=640,height=960' ! jpegenc ! rtpjpegpay ! udpsink host=Katana15.local port=8554 sync=false

# gst-launch-1.0 -v nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),fromat=NV12,width=1270,height=720' \
#     ! tee name=t ! queue ! nv3dsink sync=false \
#     t. ! queue ! nvvidconv ! "video/x-raw" ! videoconvert ! "video/x-raw,width=320,height=240,framerate=30/1" ! jpegenc ! rtpjpegpay ! udpsink host=192.168.3.9 port=8554 sync=false