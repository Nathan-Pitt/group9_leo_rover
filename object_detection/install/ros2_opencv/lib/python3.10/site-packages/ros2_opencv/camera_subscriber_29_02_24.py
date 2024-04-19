import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Global variables to easily change when needed.
UPPER_SATURATION_BOUND = 255
LOWER_SATURATION_BOUND = 100


def get_limits(colour):
    c = np.uint8([[colour]])
    hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lower_limit = hsv_c[0][0][0] - 10, 100, 100
    upper_limit = hsv_c[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit


def create_red_mask(hsv_image, lower_bound_saturation):
    lower_red_1 = np.array([0, lower_bound_saturation, 120])
    lower_red_2 = np.array([160, lower_bound_saturation, 120])

    upper_red_1 = np.array([20, UPPER_SATURATION_BOUND, 255])
    upper_red_2 = np.array([179, UPPER_SATURATION_BOUND, 255])

    mask_red_lower = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask_red_upper = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

    mask_red = mask_red_lower + mask_red_upper

    return mask_red


def highest_lower_bound_saturation(hsv_image):
    # Iterate through all values of saturation between LOWER_SATURATION_BOUND and UPPER_SATURATION_BOUND
    # return the value of saturation with the least number of contours.

    for saturation_value in range(LOWER_SATURATION_BOUND, UPPER_SATURATION_BOUND):
        contours_count = 0

        mask_red = create_red_mask(hsv_image, saturation_value)

        contours, hierarchy = cv2.findContours(
            mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Check if the contour given is 'big enough' this value can be changed to increase range, but also noise.
        for cnt in contours:
            if cnt.shape[0] > 30:  # <--- Change this integer
                contours_count += 1

        # print(f'The value of the lower bound saturation is: {i} | This value has {contours_count} contours')

        # Checks if there are any contours.
        if contours_count == 0:
            # If not then return the previous saturation value.
            min_saturation = saturation_value - 1
            return min_saturation
    return 219


def object_detection(image):
    # Convert image to HSV (Hue, Saturation, Value) image to better map it
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound_saturation = highest_lower_bound_saturation(hsv_image)
    print(f'The last saturation value with any number of contours is {lower_bound_saturation}')

    colour_object = (0, 0, 255)
    mask_red = create_red_mask(hsv_image, lower_bound_saturation)

    contours, hierarchy = cv2.findContours(
        mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if cnt.shape[0] < 30:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        x_coordinate = int(x + w / 2)
        y_coordinate = int(y + h / 2)

        image_frame = cv2.rectangle(image, (x, y),
                                    (x + w, y + h),
                                    colour_object, 2)

        cv2.putText(image_frame, "Block", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    colour_object)

        print(f"Coordinates of the red box is most likely at ({x_coordinate}, {y_coordinate})")

    cv2.imshow("object", image)
    cv2.waitKey(10)


class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.cv_bridge = CvBridge()

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        object_detection(image)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber("topic_webcam_sub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
