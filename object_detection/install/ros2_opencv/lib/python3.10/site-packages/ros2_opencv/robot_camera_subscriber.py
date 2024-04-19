import rclpy
import cv2
import numpy as np
import math

from decimal import Decimal, ROUND_HALF_UP
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# Global variables to easily change when needed.

# Bounds for saturation finding
LOWER_BOUND_VALUE = 100
UPPER_SATURATION_BOUND = 255
LOWER_SATURATION_BOUND = 130  # 140

# The tolerance range when a saturation value is found
TOLERANCE = 60

# Distance to block 2D array to get an average reading
distance_to_block = np.zeros((5, 5))
index = 0

# Check if the block to too close for a long period of time
object_close = 0


def print_text_center(text, image):
    # Function to print a specified text on screen.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (255, 255, 255)  # White color

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Centre the text on screen.
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    # Put text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)


def create_red_mask(hsv_image, lower_bound_saturation, upper_bound_saturation=UPPER_SATURATION_BOUND):
    # Gets the lower and upper bounds of the saturation value and returns a mask for the colour red.
    lower_red_1 = np.array([0, lower_bound_saturation, LOWER_BOUND_VALUE])
    lower_red_2 = np.array([160, lower_bound_saturation, LOWER_BOUND_VALUE])

    upper_red_1 = np.array([20, upper_bound_saturation, 255])
    upper_red_2 = np.array([179, upper_bound_saturation, 255])

    mask_red_lower = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask_red_upper = cv2.inRange(hsv_image, lower_red_2, upper_red_2)

    mask_red = mask_red_lower + mask_red_upper

    return mask_red


def location_of_red_block(hsv_image):
    # Iterate through all values of saturation between LOWER_SATURATION_BOUND and UPPER_SATURATION_BOUND
    # return the x,y pixel position coordinate of the centre of the block.
    x_coordinate = -1
    y_coordinate = -1

    for saturation_value in range(LOWER_SATURATION_BOUND, UPPER_SATURATION_BOUND):
        contours_count = 0

        mask_red = create_red_mask(hsv_image, saturation_value)

        contours, hierarchy = cv2.findContours(
            mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Check if the contour given is 'big enough' this value can be changed to increase range, but also noise.
        for cnt in contours:
            if cnt.shape[0] > 30:  # <--- Change this integer
                contours_count += 1

        # Checks if there are any contours.
        if contours_count != 0:
            # If yes then assign x,y values ready for return
            (x, y, w, h) = cv2.boundingRect(contours[0])
            x_coordinate = int(x + w / 2)
            y_coordinate = int(y + h / 2)
        else:
            # If no contours return the latest assigned x,y coordinate.
            break
    return x_coordinate, y_coordinate


def get_mask_bounds(hsv_image, x, y, tolerance=TOLERANCE):
    # Extract HSV values from the pixel
    hsv_pixel = hsv_image[y][x]
    hue, saturation, value = hsv_pixel
    # print(f'HSV values of selected pixel are:- Hue: {hue} | Saturation: {saturation} | Value: {value} ')

    # Define lower and upper bounds
    lower_bound = np.array([max(0, hue - tolerance), max(0, saturation - tolerance), max(0, value - tolerance)])
    upper_bound = np.array([min(179, hue + tolerance), min(255, saturation + tolerance), min(255, value + tolerance)])

    return lower_bound, upper_bound


def object_detection(image):
    # Convert image to hsv (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get x, y coordinate of pixel of block.
    x_coordinate, y_coordinate = location_of_red_block(hsv_image)

    if x_coordinate == -1:
        # If no block was found, print 'No Block Found' in the center of the screen.
        print_text_center("No Block Found", image)
    else:
        # Get new bounds and create red mask.
        lower_bound, upper_bound = get_mask_bounds(hsv_image, x_coordinate, y_coordinate)

        mask_red = cv2.inRange(hsv_image, lower_bound, upper_bound)

        contours, hierarchy = cv2.findContours(
            mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Double check if any objects was detected.
        if len(contours) > 0:

            # Get the largest object detected to help cancel out noise.
            largest_contour = max(contours, key=cv2.contourArea)

            # Get size values and coordinates
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            x_coordinate = int(x + w // 2)
            y_coordinate = int(y + h)  # int(y + h // 2)

            if y_coordinate == 720:
                y_coordinate = 719

            # Print rectangle to screen
            rect_frame = cv2.rectangle(image, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            # Print text to screen
            cv2.putText(rect_frame, "Block", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

            # print(f"Coordinates of the red box is most likely at ({x_coordinate}, {y_coordinate})")

            # Print center of the block dot.
            circle_frame = cv2.circle(image, (x_coordinate, y_coordinate), 5, (0, 255, 0), -1)

            coordinates_text = f"X: {x_coordinate} | Y: {y_coordinate}"

            # Print text in top left corner.
            cv2.putText(circle_frame, coordinates_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    cv2.imshow("object", image)
    cv2.waitKey(10)


class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.rgb = None

        self.cv_bridge = CvBridge()

        # Subscribe to RGB image
        self.sub = self.create_subscription(
            msg_type=Image,
            topic='image_raw',
            # /camera/image_raw  <== robot camera topic
            # image_raw  <== webcam topic
            callback=self.rgb_callback,
            qos_profile=10)

    def rgb_callback(self, rgb_data: Image):
        self.get_logger().info('Receiving rgb video frame')
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
        self.rgb = rgb_image

        object_detection(self.rgb)


def main(args=None):
    """
    The main function.
    """
    try:
        rclpy.init(args=args)
        node = ImageSubscriber("topic_cam_sub")
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
