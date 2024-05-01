import rclpy
import cv2
import numpy as np
import pyrealsense2 as rs2
import math

import tf2_ros
import tf2_geometry_msgs

from decimal import Decimal, ROUND_HALF_UP
from rclpy.node import Node
from rclpy import time, duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, TransformStamped, Vector3
from cv_bridge import CvBridge, CvBridgeError

import os
import subprocess

# Global variables to easily change when needed.

# Bounds for saturation finding
LOWER_BOUND_VALUE = 100
UPPER_SATURATION_BOUND = 255
LOWER_SATURATION_BOUND = 140  # 140

# Distance from the ground to the camera
camera_to_ground = 20

# The tolerance range when a saturation value is found
TOLERANCE = 50  # 60

# Distance to block 2D array to get an average reading
distance_to_block = np.zeros((5, 5))
index = 0

# Check if the block to too close / stable for a long period of time
object_close = 0

block_stable = 0
stable_block_z = 0
stable_block_x = 0

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
            if cnt.shape[0] > 1:  # <--- Change this integer | smaller = smaller object detection | default: 30
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


def object_detection(image, depth, intrinsics):
    # Convert image to HSV (Hue, Saturation, Value) image to better map it
    global distance_to_block
    global object_close
    global index
    global camera_to_ground

    # Initialising variables to return
    x_target = 0.0
    y_target = 0.0
    z_target = 0.0

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

            if y_coordinate == 480:
                y_coordinate = 479

            # Print rectangle to screen
            rect_frame = cv2.rectangle(image, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            # Print text to screen
            cv2.putText(rect_frame, "Block", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

            # print(f"Coordinates of the red box is most likely at ({x_coordinate}, {y_coordinate})")

            # =========================== Depth Code ============================
            # Assign coordinates to depth indexes
            v = int(y_coordinate)
            u = int(x_coordinate)

            # Check if distance to block is 0 (should never be 0)
            if depth[v][u] == 0:
                object_close += 1
            else:
                # Get the distance to a grouping of pixels around center
                tolerance = 2

                try:
                    distance_to_block[index][0] = depth[v][u]
                    distance_to_block[index][1] = depth[v + tolerance][u]
                    distance_to_block[index][2] = depth[v - tolerance][u]
                    distance_to_block[index][3] = depth[v][u + tolerance]
                    distance_to_block[index][4] = depth[v][u - tolerance]
                except IndexError:
                    print("One of the indexes was out of range")

                # Reset the object close detector
                object_close = 0

                # Increment index but not to equal 5
                index = (index + 1) % 5
                # print(f"Depth: {depth[v][u]} | Average Distance: {np.mean(distance_to_block)}")

            # Convert the 2D array of distance to block to new array of averages of each row
            average_distance = np.mean(distance_to_block, axis=1)

            # Sort the new array
            average_distance = np.sort(average_distance)

            # Discard the first and last elements
            distance = average_distance[1:-1]

            # Angle of camera relative to ground
            theta = 45

            # Calculate the x, y, z coordinate of object relative to camera
            x_temp = np.mean(distance) * ((u - intrinsics.ppx) / intrinsics.fx)
            y_temp = np.mean(distance) * ((v - intrinsics.ppy) / intrinsics.fy)
            z_temp = np.mean(distance)  # np.sqrt(np.abs((390**2) - np.mean(distance)**2))

            # print(f'x_temp: {x_temp} | y_temp: {y_temp} | z_temp: {z_temp}')

            # Do additional math such as centering offset and angle of camera
            x_target = x_temp - 35
            # 35 is RGB camera module offset from the center of the realsense
            y_target = -(z_temp * math.sin(theta) + y_temp * math.cos(theta))
            # z_target = z_temp * math.cos(theta) + y_temp * math.sin(theta)

            # Try and use pythagoras to calculate actual z distance.
            try:
                z_target = math.sqrt(z_temp ** 2 - camera_to_ground ** 2)
            except ValueError as e:
                print(e)
                z_target = z_temp

            coordinates_text = "(" + str(Decimal(str(x_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                               ", " + str(Decimal(str(y_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                               ", " + str(Decimal(str(z_target)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

            # See if object is too close, can change value.
            if object_close > 10:
                print_text_center("Block too Close", image)

            else:
                # Print center of the block dot.
                circle_frame = cv2.circle(image, (x_coordinate, y_coordinate), 5, (0, 255, 0), -1)

                # Print text in top left corner.
                cv2.putText(circle_frame, coordinates_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    cv2.imshow("object", image)
    cv2.waitKey(10)

    return float(x_target), float(y_target), float(z_target)


class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.block_location = Point()
        self.block_location.x = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cv_bridge = CvBridge()

        transform_camera_to_robot_base = TransformStamped()
        transform_camera_to_robot_base.header.stamp = self.get_clock().now().to_msg()
        transform_camera_to_robot_base.header.frame_id = 'px150/base_link'
        transform_camera_to_robot_base.child_frame_id = 'realsense/camera_frame'
        transform_camera_to_robot_base.transform.translation = Vector3(x=0.0,
                                                                       y=0.0,
                                                                       z=0.0)
        # x=0.0, y=0.7, z=0.58

        self.tf_broadcaster.sendTransform(transform_camera_to_robot_base)

        # Subscribe to RGB image
        self.sub = self.create_subscription(
            msg_type=Image,
            topic='/camera/color/image_raw',
            # /camera/color/image_raw  <== depth camera rgb topic
            # image_raw  <== webcam topic
            callback=self.rgb_callback,
            qos_profile=10)

        # Subscribe to get intrinsics of camera
        self.sub_info = self.create_subscription(
            msg_type=CameraInfo,
            topic='/camera/aligned_depth_to_color/camera_info',
            callback=self.image_depth_info_callback,
            qos_profile=10)

        # Subscribe to depth image
        self.sub_depth = self.create_subscription(
            msg_type=Image,
            topic='/camera/aligned_depth_to_color/image_raw',
            # /camera/depth/image_rect_raw  <== depth camera topic
            callback=self.depth_callback,
            qos_profile=10)

        self.pub_block_location = self.create_publisher(
            msg_type=Point,
            topic='/block/location',
            qos_profile=10)
        self.timer = self.create_timer(0.5, self.publish_block_location)

    def publish_block_location(self):
        global block_stable
        global stable_block_z
        global stable_block_x

        self.pub_block_location.publish(self.block_location)
        #self.get_logger().info(f'Publishing block location of: X: {self.block_location.x} | Y: {self.block_location.y}' # <=================================
        #                       f' | Z: {self.block_location.z}')

        # Broadcast transform between camera and object frame
        transform_object_to_camera = TransformStamped()
        transform_object_to_camera.header.stamp = self.get_clock().now().to_msg()
        transform_object_to_camera.header.frame_id = 'realsense/camera_frame'
        transform_object_to_camera.child_frame_id = 'object_frame'
        transform_object_to_camera.transform.translation = Vector3(x=-(self.block_location.x/1000),
                                                                   y=-(self.block_location.z/1000),
                                                                   z=-(camera_to_ground/1000))
        self.tf_broadcaster.sendTransform(transform_object_to_camera)

        frame1 = 'px150/base_link'
        frame2 = 'object_frame'

        try:
            wait = self.tf_buffer.can_transform(frame1, frame2, rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
            position = self.tf_buffer.lookup_transform(frame1, frame2, rclpy.time.Time())

            # self.get_logger().info(str(position.transform.translation))                                                # <=================================

            block_x = position.transform.translation.x
            block_y = position.transform.translation.y
            block_z = position.transform.translation.z

            print(f"block_x: {block_x}\nblock_y: {block_y}\nblock_z: {block_z}")

            # Check if block is within range of robot arm
            if 0.1 > block_x > -0.1 and 0.5 > block_z > -0.5:
                print("BLOCK FOUND WITHIN RANGE OF ARM")

                # Check if the block is stable and not moving (within 5mm)
                if abs(block_x - stable_block_x) <= (5 * 1000) and abs(block_z - stable_block_z) <= (5 * 1000):
                    block_stable += 1
                    print(f"BLOCK REMAINED STABLE FOR: {block_stable}")
                else:
                    print("BLOCK NOT STABLE, RESET STABLE VARIABLE")
                    block_stable = 0

                # Check if after 10 tics of the publisher if the block has remained stable.
                if block_stable > 10:
                    params = [str(block_x),
                              str(block_y),
                              str(block_z)]

                    script_path = os.path.expanduser('~/test/test.py')
                    subprocess.Popen(['python3', script_path] + params)

                    block_stable = -20

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().info("Didn't TF")

    def rgb_callback(self, rgb_data: Image):
        # self.get_logger().info('Receiving rgb video frame')                                                            # <=================================
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
        self.rgb = rgb_image

        # Check if I have intrinsic and depth info before starting object detection
        if self.depth is not None and self.intrinsics is not None:
            self.block_location.x, self.block_location.y, self.block_location.z = object_detection(self.rgb, self.depth,
                                                                                                   self.intrinsics)
        elif self.depth is None:
            print("Waiting on depth image")
        elif self.intrinsics is None:
            print("Waiting on intrinsics")

    def depth_callback(self, depth_data: Image):
        # self.get_logger().info('Receiving depth video frame')                                                          # <=================================
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_data, depth_data.encoding)
        self.depth = depth_image

    def image_depth_info_callback(self, camera_info: CameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = camera_info.width
            self.intrinsics.height = camera_info.height
            self.intrinsics.ppx = camera_info.k[2]
            self.intrinsics.ppy = camera_info.k[5]
            self.intrinsics.fx = camera_info.k[0]
            self.intrinsics.fy = camera_info.k[4]
            self.intrinsics.model = rs2.distortion.none
            self.intrinsics.coeffs = [i for i in camera_info.d]
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
