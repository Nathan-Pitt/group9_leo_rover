import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP

# import pyrealsense2.pyrealsense2 as rs2

# Global variables to easily change when needed.
UPPER_SATURATION_BOUND = 255
LOWER_SATURATION_BOUND = 130  # 140
distance_to_block = 0
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
    lower_red_1 = np.array([0, lower_bound_saturation, 100])
    lower_red_2 = np.array([160, lower_bound_saturation, 100])

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


def get_mask_bounds(hsv_image, x, y, tolerance=50):
    # Extract HSV values from the pixel
    hsv_pixel = hsv_image[y][x]
    hue, saturation, value = hsv_pixel
    print(f'HSV values of selected pixel are:- Hue: {hue} | Saturation: {saturation} | Value: {value} ')

    # Define lower and upper bounds
    lower_bound = np.array([max(0, hue - tolerance), max(0, saturation - tolerance), max(0, value - tolerance)])
    upper_bound = np.array([min(179, hue + tolerance), min(255, saturation + tolerance), min(255, value + tolerance)])

    return lower_bound, upper_bound


def object_detection(image, depth, intrinsics):
    # Convert image to HSV (Hue, Saturation, Value) image to better map it
    global distance_to_block
    global object_close

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
            y_coordinate = int(y + h // 2)

            # Print rectangle to screen
            rect_frame = cv2.rectangle(image, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            # Print text to screen
            cv2.putText(rect_frame, "Block", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

            print(f"Coordinates of the red box is most likely at ({x_coordinate}, {y_coordinate})")

            # =========================== Depth Code ============================
            # Assign coordinates to depth indexes
            v = int(y_coordinate)
            u = int(x_coordinate)

            # Check if distance to block is 0 (should never be 0)
            if depth[v][u] == 0:
                object_close += 1
            else:
                distance_to_block = depth[v][u]
                object_close = 0
            print(f"Depth: {depth[v][u]}")
            # Convert distance from mm to m
            # if distance_to_block != 0:
            #    distance_to_block = distance_to_block/1000

            theta = 0

            print(f"intrinsics: ppx: {intrinsics.ppx} | ppy: {intrinsics.ppy}")
            print(f"fx: {intrinsics.fx} | fy: {intrinsics.fy}")
            print(f"distance to block: {distance_to_block}")

            x_temp = distance_to_block * ((u - intrinsics.ppx) / intrinsics.fx)
            y_temp = distance_to_block * ((v - intrinsics.ppy) / intrinsics.fy)
            z_temp = distance_to_block

            x_target = x_temp - 35
            # 35 is RGB camera module offset from the center of the realsense
            y_target = -(z_temp * math.sin(theta) + y_temp * math.cos(theta))
            z_target = z_temp * math.cos(theta) + y_temp * math.sin(theta)

            print(f"z_target: {z_target} | x_target: {x_target} | y_target: {y_target}")
            #try:
            #    z_target_final = math.sqrt((z_target**2 - x_target**2) - y_target**2)
            #except ValueError as e:
            #    print(e)
            #    z_target_final = z_target

            coordinates_text = "(" + str(Decimal(str(x_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                               ", " + str(Decimal(str(y_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                               ", " + str(Decimal(str(z_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

            if object_close > 10:
                print_text_center("Block too Close", image)

            else:
                circle_frame = cv2.circle(image, (x_coordinate, y_coordinate), 5, (0, 255, 0), -1)

                cv2.putText(circle_frame, coordinates_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    cv2.imshow("object", image)
    cv2.waitKey(10)


class CameraIntrinsics:
    def __init__(self, ppx, ppy, fx, fy):
        self.ppx = ppx
        self.ppy = ppy
        self.fx = fx
        self.fy = fy


class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.rgb = None
        self.depth = None
        self.intrinsics = None

        self.sub = self.create_subscription(
            msg_type=Image,
            topic='/camera/color/image_raw',
            # /camera/color/image_raw  <== depth camera rgb topic
            # image_raw  <== webcam topic
            callback=self.rgb_callback,
            qos_profile=10)

        self.sub_info = self.create_subscription(
            msg_type=CameraInfo,
            topic='/camera/depth/camera_info',
            callback=self.image_depth_info_callback,
            qos_profile=10)

        self.sub_depth = self.create_subscription(
            msg_type=Image,
            topic='/camera/depth/image_rect_raw',
            # /camera/depth/image_rect_raw  <== depth camera topic
            callback=self.depth_callback,
            qos_profile=10)

        self.cv_bridge = CvBridge()

    def rgb_callback(self, rgb_data: Image):
        self.get_logger().info('Receiving rgb video frame')
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
        self.rgb = rgb_image
        if self.depth is not None and self.intrinsics is not None:
            object_detection(self.rgb, self.depth, self.intrinsics)

    def depth_callback(self, depth_data: Image):
        self.get_logger().info('Receiving depth video frame')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_data, depth_data.encoding)
        self.depth = depth_image

    def image_depth_info_callback(self, camera_info: CameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = CameraIntrinsics(ppx=camera_info.k[2],
                                               ppy=camera_info.k[5],
                                               fx=camera_info.k[0],
                                               fy=camera_info.k[4])
        except CvBridgeError as e:
            print(e)
            return


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
