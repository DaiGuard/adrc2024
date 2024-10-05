import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

import os
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

import numpy as np
import matplotlib.pyplot as plt

class StreamTestNode(Node):

    def __init__(self):
        super().__init__('stream_test')

        GObject.threads_init()
        Gst.init(None)

        launch_str  = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=15/1 ! "
        launch_str += "nvvideoconvert ! video/x-raw,width=640,height=480 ! "
        launch_str += "gdkpixbufoverlay location=data/mask.png ! "
        launch_str += "videoconvert ! video/x-raw,format=RGB ! "
        launch_str += "appsink async=true name=sink"

        self.pipeline = Gst.parse_launch(launch_str)
        if not self.pipeline:
            raise RuntimeError("unable to create pipeline")
        
        sink = self.pipeline.get_by_name("sink")
        if not sink:
            raise RuntimeError("unable to get sink element")
        
        sinkpad = sink.get_static_pad("sink")
        if not sinkpad:
            raise RuntimeError("unable to get sink pad")
        
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe, 0)

        self.loop = GObject.MainLoop()

    def sink_pad_buffer_probe(self, pad, info, data):
        caps = pad.get_current_caps()
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.DROP
        
        result, map_info = gst_buffer.map(Gst.MapFlags.READ)
        if result:
            image = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=map_info.data)

        return Gst.PadProbeReturn.OK

    def start_pipeline(self):
        """パイプライン開始
        """
        self.get_logger().info("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            rclpy.spin(self)
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)