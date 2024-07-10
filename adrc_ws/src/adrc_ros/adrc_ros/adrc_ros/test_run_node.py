import rclpy
import rclpy.logging
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import torch
import torchvision
import torchvision.transforms as transforms

from torch2trt import torch2trt

class TestRunNode(Node):

    def __init__(self):
        super().__init__("test_run")

        # パラメータ登録
        self.model_path = "models/model.pth"

        self.declare_parameter("model_path", self.model_path)
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value

        # ROSパラメータサーバの登録
        self.add_on_set_parameters_callback(self.parameters_cb)

        # モデルの準備
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
        
        # Cv変換の準備
        self.cv_bridge = CvBridge()

        # コマンド配信準備
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # データ購読登録
        self.image_sub = self.create_subscription(Image, "/front_camera/raw", self.image_cb, 10)

    def parameters_cb(self, params):
        for param in params:
            if param.name == 'model_path':
                self.model_path = param.value

        return SetParametersResult(successful=True)

    def image_cb(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        input_tensor = self.transform(cv_image)
        input_batch = input_tensor.unsqueeze(0)
        output = self.model(input_batch.to(self.device))
        output = output.to('cpu')

        cmd = Twist()
        cmd.linear.x = float(output[0][0])
        cmd.angular.z = float(output[0][1])

        self.cmd_vel_pub.publish(cmd)
