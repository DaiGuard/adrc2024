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
        self.mask_file = "mask.png"

        self.declare_parameter("model_path", self.model_path)
        self.declare_parameter("mask_file", self.mask_file)
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.mask_file = self.get_parameter("mask_file").get_parameter_value().string_value

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

        # マスク画像の読み込み
        self.mask = cv2.imread(self.mask_file, cv2.IMREAD_UNCHANGED)
        
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
            if param.name == 'mask_file':
                self.mask_file = param.value

        return SetParametersResult(successful=True)

    def image_cb(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        cv_image[:, :] = cv_image[:, :] * (1 - self.mask[:,:,3:] / 255) \
                    + cv_image[:,:,:3] * self.mask[:,:,3:] / 255            

        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        image = self.transform(cv_image)
        image = image.unsqueeze(dim=0)
        image = image.to(self.device)
        output = self.model(image)
        output = output.to('cpu')

        cmd = Twist()
        # cmd.linear.x = float(output[0][0])
        # cmd.angular.z = float(output[0][1])
        cmd.linear.x = 0.4
        cmd.angular.z = - float(output[0])

        self.cmd_vel_pub.publish(cmd)
