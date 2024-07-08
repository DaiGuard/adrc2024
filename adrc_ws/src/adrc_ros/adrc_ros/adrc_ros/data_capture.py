import rclpy
from adrc_ros.data_capture_node import DataCaptureNode

def main(args=None):
    rclpy.init(args=args)

    node = DataCaptureNode()

    rclpy.spin(node)

    node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
