import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, JointState
from geometry_msgs.msg import Twist


# =================== SUBSCRIBER ===================
class LeoRoverControllerNode(Node):
    """A ROS2 node that subscribes to /scan and /camera/depth/image_raw to gather nessasary info
    for the leo rover behaviour"""

    def __init__(self):
        super().__init__('leo_rover_controller_node')
        self.scan_subscriber_node = self.create_subscription(
            msg_type=LaserScan,
            topic='/scan',
            callback=self.scan_subscriber_callback,
            qos_profile=1)

        self.camera_depth_subscriber_node = self.create_subscription(
            msg_type=Image,
            topic='/camera/depth/image_rect_raw',
            callback=self.depth_subscriber_callback,
            qos_profile=1)

        self.camera_color_subscriber_node = self.create_subscription(
            msg_type=Image,
            topic='/camera/color/image_raw',
            callback=self.color_subscriber_callback,
            qos_profile=1)

        self.cmd_vel_publisher = self.create_publisher(
            msg_type=Twist,
            topic='/cmd_vel',
            qos_profile=1)

        self.arm_state_publisher = self.create_publisher(
            msg_type=JointState,
            topic='/px150/joint_states',
            qos_profile=1)

        timer_period: float = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def scan_subscriber_callback(self, msg: LaserScan):
        """Function that is periodically called by the timer."""
        self.get_logger().info(f"""
        I have received the scan information.
        """)

    def depth_subscriber_callback(self, msg: Image):
        """Function that is periodically called by the timer."""
        self.get_logger().info(f"""
        I have received the depth information.
        """)

    def color_subscriber_callback(self, msg: Image):
        """Function that is periodically called by the timer."""
        self.get_logger().info(f"""
        I have received the color information.
        """)

    def timer_callback(self):
        """Function that is periodically called by the timer."""

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.5

        self.cmd_vel_publisher.publish(cmd_vel_msg)


def main(args=None):
    """Main function"""

    try:
        rclpy.init(args=args)

        leo_rover_controller_node = LeoRoverControllerNode()

        rclpy.spin(leo_rover_controller_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
