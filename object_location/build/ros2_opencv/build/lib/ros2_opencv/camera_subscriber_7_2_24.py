import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Specify upper and lower bounds for the colours
lower_red_1 = np.array([0, 110, 120])
upper_red_1 = np.array([20, 255, 255])

lower_red_2 = np.array([160, 110, 120])
upper_red_2 = np.array([179, 255, 255])

class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.sub = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.cv_bridge = CvBridge()

    def object_detect(self, image):
        # Convert image to HSV (Hue, Saturation, Value) image to better map it
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create the masks and add them together if needed
        mask_red_lower = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask_red_upper = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

        mask_red = mask_red_lower + mask_red_upper

        contours, hierarchy = cv2.findContours(
            mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cnt.shape[0] < 15:
                continue

            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(image, (int(x+w/2), int(y+h/2)), 5,
                       (0, 255, 0), -1)

        cv2.imshow("object", image)
        cv2.waitKey(10)

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        self.object_detect(image)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber("topic_webcam_sub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
