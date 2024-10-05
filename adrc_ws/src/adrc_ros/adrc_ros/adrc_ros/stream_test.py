import rclpy
from adrc_ros.stream_test_node import StreamTestNode

def main(args=None):
    rclpy.init(args=args)

    node = StreamTestNode()

    node.start_pipeline()
    # rclpy.spin(node)

    node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
