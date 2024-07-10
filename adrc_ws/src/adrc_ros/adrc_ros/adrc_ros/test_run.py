import rclpy
from adrc_ros.test_run_node import TestRunNode

def main(args=None):
    rclpy.init(args=args)

    node = TestRunNode()

    rclpy.spin(node)

    node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
