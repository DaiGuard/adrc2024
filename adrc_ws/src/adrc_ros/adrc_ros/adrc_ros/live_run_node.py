import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

import os
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

import pyds

from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool
from cv_bridge import CvBridge

import torch
import torchvision.transforms as transforms
from torch2trt import torch2trt

import time
import numpy as np
import cv2
import math


class LiveRunNode(Node):

    def __init__(self):
        super().__init__('live_run')

        GObject.threads_init()
        Gst.init(None)

        # Gstreamerランチャの宣言
        launch_str = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=15/1 ! m.sink_0 "
        # launch_str+= "nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=60/1 ! nvvideoconvert flip-method=rotate-180 ! m.sink_1 "
        launch_str+= "nvstreammux name=m width=1280 height=720 batch-size=1 num-surfaces-per-frame=1 "
        # launch_str+= "nvstreammux name=m width=1280 height=1440 batch-size=2 num-surfaces-per-frame=1 "
        launch_str+= "! nvmultistreamtiler columns=1 rows=1 width=1280 height=720 "
        # launch_str+= "! nvmultistreamtiler columns=1 rows=2 width=1280 height=1440 "
        launch_str+= "! nvvideoconvert ! video/x-raw(memory:NVMM),width=640,height=360,format=RGBA ! fakesink name=sink sync=false "
        #launch_str+= "! nvvideoconvert ! video/x-raw(memory:NVMM),width=640,height=720,format=RGBA ! tee name=t ! queue ! fakesink name=sink sync=false "
        # launch_str+= "t.src_1 ! queue ! nvvidconv ! video/x-raw,width=640,height=720 ! jpegenc ! rtpjpegpay ! udpsink host=192.168.3.103 port=8554 sync=false"
        self.pipeline = Gst.parse_launch(launch_str)
        if not self.pipeline:
            raise RuntimeError('unable to create pipeline')
        
        # Gstreamerプローブの取得
        sink = self.pipeline.get_by_name("sink")
        if not sink:
            raise RuntimeError('unable to get sink element')
        sinkpad = sink.get_static_pad("sink")
        if not sinkpad:
            raise RuntimeError('unable to get sink pad')
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe, 0)

        # Gstreamerバス状態の監視
        self.loop = GObject.MainLoop()

        # ROSパラメータ登録
        self.model_path = 'models/model_weight.pth'
        self.mask_file = "mask.png"

        self.declare_parameter('model_path', self.model_path)
        self.declare_parameter("mask_file", self.mask_file)        

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.mask_file = self.get_parameter("mask_file").get_parameter_value().string_value        

        # ROSトピック購読登録
        self.state_vel_sub = self.create_subscription(
            Int32, '/status', self.status_cb, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/pose', self.pose_cb, 10
        )
        self.curvel_sub = self.create_subscription(
            Twist, '/cur_vel', self.curvel_cb, 10
        )
        self.cmdvel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.preview_img_pub = self.create_publisher(
            CompressedImage,
            '/preview',
            10
        )

        # ROSパラメータ購読登録
        self.add_on_set_parameters_callback(self.parameters_cb)

        # 共有用データ
        self.bridge = CvBridge()
        self.current_vel = [ 0.0, 0.0 ]
        self.current_pose = [ 0.0, 0.0, 0.0 ]
        self.cmd_vel = [ 0.0, 0.0 ]

        self.device = torch.device('cuda')
        self.transform = transforms.Compose([            
            transforms.ToPILImage(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model = torch.load(self.model_path)
        model = model.to(self.device)
        # data = torch.zeros((1, 3, 224, 224)).to(device)
        # self.model_trt = torch2trt(model, [data], fp16_mode=True)
        self.model = model.eval()

    def parameters_cb(self, params):
        for param in params:
            print("set ", param)
            if param.name == 'model_path':
                self.model_path = param.value

        return SetParametersResult(successful=True)


    def status_cb(self, msg: Int32):
        # print('status', msg)
        pass

    def pose_cb(self, msg: PoseStamped):
        self.current_pose[0] = msg.pose.position.x
        self.current_pose[1] = msg.pose.position.y
        self.current_pose[2] = math.acos(msg.pose.orientation.w) * 2.0

    def curvel_cb(self, msg: Twist):
        self.current_vel[0] = msg.linear.x
        self.current_vel[1] = msg.angular.z

    def sink_pad_buffer_probe(self, pad, info, data):
        caps = pad.get_current_caps()
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.DROP
    
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            ######
            image = cv2.rectangle(image, (0, 0), (640, 130), (0, 0, 0), thickness=-1)
            ######
            image = cv2.rectangle(image, (150, 260), (490, 360), (0, 0, 0), thickness=-1)
            input_tensor = self.transform(image)            
            input_batch = input_tensor.unsqueeze(0)
            output = self.model(input_batch.to(self.device))
            output = output.to('cpu')

            # u = int(image.shape[1] / 2.0 * (1.0 - output[0][1]))
            # v = int(image.shape[0] / 2.0 * (1.0 - output[0][0]))
            # image = cv2.circle(image, (u, v), 10, (255,0,0), thickness=-1)

            # imgmsg = self.bridge.cv2_to_compressed_imgmsg(image)
            # self.preview_img_pub.publish(imgmsg)

            cmd_vel = Twist()
            max_throttle = 0.09
            min_throttle = 0.06
            throttle = 0.0
            # steer = float(output[0][1])
            steer = float(output[0])

            if math.fabs(steer) > 0.5:
                steer = 0.5 *  steer / math.fabs(steer)
            else:
                steer = steer

            throttle = (max_throttle - min_throttle) * (1.0 - math.fabs(steer) / 0.5) + min_throttle
            cmd_vel.linear.x = throttle
            cmd_vel.linear.y = 0.0
            cmd_vel.linear.z = 0.0
            cmd_vel.angular.x = 0.0
            cmd_vel.angular.y = 0.0
            cmd_vel.angular.z = steer

            # if float(output[0][0]) > 0.8:
            #     cmd_vel.linear.x = 0.8
            # else:
            #     cmd_vel.linear.x = float(output[0][0])
            # cmd_vel.linear.y = 0.0
            # cmd_vel.linear.z = 0.0
            # cmd_vel.angular.x = 0.0
            # cmd_vel.angular.y = 0.0
            # if math.fabs(float(output[0][1])) > 0.5:
            #     cmd_vel.angular.z = 0.5 * float(output[0][1]) / math.fabs(float(output[0][1]))
            # elif math.fabs(float(output[0][1])) > 0.35:
            #     cmd_vel.angular.z = 0.35 * float(output[0][1]) / math.fabs(float(output[0][1]))
            # elif math.fabs(float(output[0][1])) > 0.1:
            #     cmd_vel.angular.z = 0.1 * float(output[0][1]) / math.fabs(float(output[0][1]))
            # else:
            #     cmd_vel.angular.z = float(output[0][1])
            self.cmdvel_pub.publish(cmd_vel)

            try:
                l_frame=l_frame.next
            except StopIteration:
                break            

        return Gst.PadProbeReturn.OK
    
    def start_pipeline(self):
        """パイプライン開始
        """
        self.get_logger().info("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            rclpy.spin(self)
            # self.loop.run()
        except:
            pass

        self.pipeline.set_state(Gst.State.NULL)