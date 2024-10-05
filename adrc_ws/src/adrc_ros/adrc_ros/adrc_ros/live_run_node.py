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

        # ROSパラメータ登録
        self.model_path = 'models/model_weight.pth'
        self.mask_file = "data/mask.png"

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

        # マスク画像の読み込み
        self.mask = cv2.imread(self.mask_file, cv2.IMREAD_UNCHANGED)

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
        self.model = model.eval()

        launch_str  =  'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),fromat=NV12,width=1280,height=720,framerate=15/1 ! '
        launch_str +=  'nvvideoconvert ! video/x-raw,width=640,height=480 ! '
        launch_str += f'gdkpixbufoverlay location={self.mask_file} ! '
        launch_str +=  'videoconvert ! video/x-raw,format=RGB ! '
        launch_str +=  'appsink async=true name=sink'

        # Gstreamerランチャの宣言
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


    def parameters_cb(self, params):
        print(f'param set: {params}')
        for param in params:
            if param.name == 'model_path':
                self.model_path = param.value
            if param.name == 'mask_file':
                self.mask_file = param.value

        return SetParametersResult(successful=True)


    def status_cb(self, msg: Int32):
        # print(f'status: {msg.data}')
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
        
        result, map_info = gst_buffer.map(Gst.MapFlags.READ)
        if result:
            image = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=map_info.data)
            input_tensor = self.transform(image)            
            input_batch = input_tensor.unsqueeze(0)
            output = self.model(input_batch.to(self.device))
            output = output.to('cpu')

            cmd_vel = Twist()
            max_throttle = 0.09
            min_throttle = 0.06
            throttle = 0.0
            steer = - float(output[0])

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

            self.cmdvel_pub.publish(cmd_vel)

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