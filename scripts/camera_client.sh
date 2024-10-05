#! /bin/env bash

gst-launch-1.0 udpsrc port=8554 ! "application/x-rtp,encoding-name=JPEG" ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink