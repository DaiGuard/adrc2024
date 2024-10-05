import rclpy
from adrc_ros.live_run_node import LiveRunNode

def main(args=None):
    rclpy.init(args=args)

    node = LiveRunNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass