import rclpy
import rclpy.logging
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, LaserScan
from std_srvs.srv import SetBool

import os
import pathlib
import uuid
import cv2
from cv_bridge import CvBridge

class DataCaptureNode(Node):

    def __init__(self):
        super().__init__("data_capture")

        self.recording = False
        self.last_timestamp = 0.0

        # パラメータ登録
        self.record_timespan = 1.0
        self.record_path = "data/"

        self.declare_parameter("timespan", self.record_timespan)
        self.record_timespan = self.get_parameter("timespan").get_parameter_value().double_value
        self.declare_parameter("path", self.record_path)
        self.record_path = self.get_parameter("path").get_parameter_value().string_value

        # ROSパラメータサーバの登録
        self.add_on_set_parameters_callback(self.parameters_cb)

        # データ購読登録
        self.current_vel_sub = Subscriber(self, TwistStamped, "/current_vel")
        self.front_image_sub = Subscriber(self, Image, "/front_camera/raw")
        self.rear_image_sub = Subscriber(self, Image, "/rear_camera/raw")
        self.scan_sub = Subscriber(self, LaserScan, "/scan" )

        self.msg_sync = ApproximateTimeSynchronizer(
            [self.current_vel_sub, self.front_image_sub, self.rear_image_sub, self.scan_sub],
            10, 0.5)
        self.msg_sync.registerCallback(self.message_cb)

        # サービス登録
        self.create_service(SetBool, "~/record", self.record_cb)

        # デバック画像配信
        self.cv_bridge = CvBridge()
        self.debug_image_pub = self.create_publisher(Image, "~/debug", 10)


    def parameters_cb(self, params):
        for param in params:
            if param.name == 'timespan':
                self.record_timespan = param.value
            elif param.name == "path":
                self.record_path = param.value

        return SetParametersResult(successful=True)
    
    def message_cb(self, current_vel, front_image, rear_image, scan):
        
        cv_front = self.cv_bridge.imgmsg_to_cv2(front_image)
        cv_rear = self.cv_bridge.imgmsg_to_cv2(rear_image)

        cv_debug = cv2.hconcat([cv_front, cv_rear])
        cv2.putText(cv_debug, "{}, {}".format(current_vel.twist.linear.x, current_vel.twist.angular.z),
                    (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
        if self.recording:
            cv2.putText(cv_debug, "REC",
                        (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))        
        debug_image = self.cv_bridge.cv2_to_imgmsg(cv_debug)        

        current_time = current_vel.header.stamp.sec + current_vel.header.stamp.nanosec * 10e-9
        if self.recording:
            if current_time - self.last_timestamp > self.record_timespan:
                name = str(uuid.uuid4())

                # データセットフィルの保存
                self.dataset_file.write("{}, {}, {}\n".format(name, current_vel.twist.linear.x, - current_vel.twist.angular.z))

                # 画像ファイルの保存
                cv2.imwrite(os.path.join(self.image_path, name + ".jpg"), cv_front)
                # cv2.imwrite(os.path.join(self.image_path, name + "_front.jpg"), cv_front)
                # cv2.imwrite(os.path.join(self.image_path, name + "_rear.jpg"), cv_rear)
                
                # スキャンデータの保存
                with open(os.path.join(self.scan_path, name + ".txt"), "w") as f:
                    f.write(f"{len(scan.ranges)}, ")
                    for range in scan.ranges:
                        f.write(f"{range}, ")
                    f.write(f"{scan.angle_min}, {scan.angle_max}\n")

                self.last_timestamp = current_time

                self.debug_image_pub.publish(debug_image)                
        else:
            self.debug_image_pub.publish(debug_image)

    def record_cb(self, request, response):
        if request.data:
            # フォルダの絶対パスを取得
            self.folder_path = pathlib.Path(self.record_path)
            if os.path.exists(self.folder_path):
                print("found record path: {}".format(self.folder_path))
            else:
                os.makedirs(self.folder_path)
                print("create record path: {}".format(self.folder_path))

            # イメージフォルダの絶対パスを取得
            self.image_path = os.path.join(self.folder_path, "images")
            if os.path.exists(self.image_path):
                print("found record image path: {}".format(self.image_path))
            else:
                os.mkdir(self.image_path)
                print("create record image path: {}".format(self.image_path))

            # データセットファイルの確認
            dataset_path = os.path.join(self.folder_path, "dataset.csv")
            if os.path.exists(dataset_path):
                print("found dataset file: {}".format(dataset_path))
                self.dataset_file = open(dataset_path, "a")
            else:
                print("create dataset file: {}".format(dataset_path))
                self.dataset_file = open(dataset_path, "w")

            self.scan_path = os.path.join(self.folder_path, "scans")
            if os.path.exists(self.scan_path):
                print("found record scan path: {}".format(self.scan_path))
            else:
                os.mkdir(self.scan_path)
                print("create record scan path: {}".format(self.scan_path))

        else:
            if self.dataset_file:
                self.dataset_file.close()

        self.recording = request.data

        response.success = True
        if request.data:
            response.message = "record start"
        else:
            response.message = "record stop"

        return response
