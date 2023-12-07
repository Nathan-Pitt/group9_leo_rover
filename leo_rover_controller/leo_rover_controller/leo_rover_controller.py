import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, JointState
from geometry_msgs.msg import Twist
# Author: Group 9 - Leo Rover Controller
# Version: 0.1


# =================== Initialising controller node ===================
class LeoRoverControllerNode(Node):
    """A ROS2 node that subscribes to /scan and /camera/depth/image_raw to gather nessasary info
    for the leo rover behaviour, and publishes to /cmd_vel and /px150/joint_states based on sensor readings"""

    """For now this version of the controller node is to subscribe and publish to needed topics to provide an
    accurate rqt_graph, futher behaviours will be intergrated through new methods/functions"""

    def __init__(self):
        super().__init__('leo_rover_controller_node')

        # =================== Initialising Subscribers ===================
        # Subscribe to the scan topic, published by the lidar.
        self.scan_subscriber_node = self.create_subscription(
            msg_type=LaserScan,
            topic='/scan',
            callback=self.scan_subscriber_callback,
            qos_profile=1)

        # Subscribe to the depth topic, published by the depth camera.
        self.camera_depth_subscriber_node = self.create_subscription(
            msg_type=Image,
            topic='/camera/depth/image_rect_raw',
            callback=self.depth_subscriber_callback,
            qos_profile=1)

        # Subscribe to the color topic, published by the depth camera.
        self.camera_color_subscriber_node = self.create_subscription(
            msg_type=Image,
            topic='/camera/color/image_raw',
            callback=self.color_subscriber_callback,
            qos_profile=1)

        # =================== Initialising Publishers ===================
        # Set-up publisher to send velocities to leo rover wheels.
        self.cmd_vel_publisher = self.create_publisher(
            msg_type=Twist,
            topic='/cmd_vel',
            qos_profile=1)

        # Set-up publisher to send joint states to arm.
        self.arm_state_publisher = self.create_publisher(
            msg_type=JointState,
            topic='/px150/joint_states',
            qos_profile=1)

        timer_period: float = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    # =================== Initialising Publishers ===================
    def scan_subscriber_callback(self, msg: LaserScan):
        """Function that is periodically called by the timer for scan messages from LiDAR."""
        
        self.get_logger().info(f"""
        I have received the scan information.
        """)

    def depth_subscriber_callback(self, msg: Image):
        """Function that is periodically called by the timer for depth messages from depth camera."""
        
        self.get_logger().info(f"""
        I have received the depth information.
        """)

    def color_subscriber_callback(self, msg: Image):
        """Function that is periodically called by the timer for color messages from depth camera."""
        
        self.get_logger().info(f"""
        I have received the color information.
        """)

    def timer_callback(self):
        """Function that is periodically called by the timer to publish messages to various topics."""

        # Initalise the cmd_vel message to send to the leo rover.
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.5

        # Publish message.
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
