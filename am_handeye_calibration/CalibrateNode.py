import cv2
import numpy as np
import transforms3d as t3d
import pickle

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .ImageNodeBase import ImageNodeBase
from .ArucoBoardDetector import ArucoBoardDetector


class Calibrator(ImageNodeBase):
    def __init__(self):
        super().__init__(node_name="calibrator", rgb_enable=True)

        self.detector = ArucoBoardDetector()

        # Member variables
        self.inner_corner = []
        self.img2show = None
        self.new_frame = False
        # Translation of area
        self.trans = None

        self.image_publisher = self.create_publisher(Image, "/calibrator/image", 1)
        # self.translation_publisher = self.create_publisher(Float64MultiArray, "/calibrator/translation", 3)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.marker_timer_callback)
        self.command_subscription = self.create_subscription(Int64, "/calibrator/command", self.command_callback, 1)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        # TF frame
        self.base_link = "panda_link0"
        self.eef_link = "panda_hand"
        self.camera_link = "camera_link"
        self.sensor_link = "camera_color_optical_frame"
        self.target_link = "target_link"

        # Saved transform for solving relative pose
        self.image_list = []
        self.board_R_list = []
        self.board_T_list = []
        self.board_transform = None
        self.eef_R_list = []
        self.eef_T_list = []
        self.eef_transform = None
        self.relative_transform = None
        self.relative_transform = np.array([0.0513706, -0.045675, 0.0331922, 0.699842, 0.0103786, 0.714221, -0.00154157])

    def marker_timer_callback(self):
        # translation_msg = Float64MultiArray()
        if self.rgb_flag and self.rgb_img is not None:
            self.img2show = self.rgb_img.copy()
            # Detect the markers in the image
            ret = self.detector.detect_board(self.img2show, self.camera_k, self.camera_d, show_markers=False, show_corners=False)
            # if markers can be detected
            if ret[0]:
                self.new_frame = True
                # Publish tf camera->target
                self.img2show, self.board_transform = ret[1:]
                board_transform_quat = t3d.quaternions.mat2quat(self.board_transform[:3, :3])
                board_transform_array = np.array([*self.board_transform[:3, 3],
                                                  *wxyz2xyzw(board_transform_quat)])
                self.publish_transform(board_transform_array, self.sensor_link, self.target_link)
                # Current tf base->eef
                self.eef_transform = self.get_transform(self.base_link, self.eef_link)
                # For debug
                # if self.eef_transform is not None:
                #     self.logger.info(np.array2string(self.eef_transform))

                # translation_msg.data.append(1.0)
                # translation_msg.data.extend(self.trans[0].tolist())
            else:
                # translation_msg.data.append(0.0)
                self.board_transform = None

            # Publish
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.img2show, "bgr8"))
            self.rgb_flag = False
        else:
            # translation_msg.data.append(0.0)
            pass

        if self.relative_transform is not None:
            self.publish_transform(self.relative_transform, self.eef_link, self.camera_link)

        # self.translation_publisher.publish(translation_msg)

    def command_callback(self, msg):
        command = msg.data
        if command == 1:
            if self.board_transform is not None and self.eef_transform is not None:
                board_transform = self.get_transform(self.camera_link, self.target_link)
                self.board_R_list.append(t3d.quaternions.quat2mat(xyzw2wxyz(board_transform[3:])))
                self.board_T_list.append(board_transform[:3])
                self.eef_R_list.append(t3d.quaternions.quat2mat(xyzw2wxyz(self.eef_transform[3:])))
                self.eef_T_list.append(self.eef_transform[:3])
                self.image_list.append(self.rgb_img)
            else:
                self.logger.warn("Cannot detect board pose!")

            if len(self.board_R_list) >= 5:
                relative_R, relative_T = cv2.calibrateHandEye(self.eef_R_list, self.eef_T_list, self.board_R_list, self.board_T_list,
                                                              method=cv2.CALIB_HAND_EYE_DANIILIDIS)
                relative_R = t3d.quaternions.mat2quat(relative_R)
                self.relative_transform = np.array([*relative_T[:, 0],
                                                    *wxyz2xyzw(relative_R)])
                self.logger.info(np.array2string(self.relative_transform))
            else:
                self.logger.info("Current counter: {}".format(len(self.board_R_list)))
        if command == 2:
            with open("/home/armine/ROS2/franka_ws/src/ros2_perception/am_handeye_calibration/am_handeye_calibration/calibration.pkl", 'wb') as f:
                pickle.dump({
                    'board_R_list': self.board_R_list,
                    'board_T_list': self.board_T_list,
                    'eef_R_list': self.eef_R_list,
                    'eef_T_list': self.eef_T_list,
                    'images': self.image_list
                }, f)
            self.logger.info("Save finished!")

    def get_transform(self, to_frame, from_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                time=Time(seconds=0),
                timeout=Duration(nanoseconds=5e7))
            return np.array([t.transform.translation.x,
                             t.transform.translation.y,
                             t.transform.translation.z,
                             t.transform.rotation.x,
                             t.transform.rotation.y,
                             t.transform.rotation.z,
                             t.transform.rotation.w])
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame} to {from_frame}: {ex}')
            return

    def publish_transform(self, transform, parent_frame: str, child_frame: str):
        t = TransformStamped()

        # Read message content and assign it to corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = transform[0]
        t.transform.translation.y = transform[1]
        t.transform.translation.z = transform[2]

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        t.transform.rotation.x = transform[3]
        t.transform.rotation.y = transform[4]
        t.transform.rotation.z = transform[5]
        t.transform.rotation.w = transform[6]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


def xyzw2wxyz(quat):
    return [quat[-1], *quat[:-1]]


def wxyz2xyzw(quat):
    return [*quat[1:], quat[0]]


def main(args=None):
    rclpy.init(args=args)

    image_node_base = Calibrator()

    rclpy.spin(image_node_base)

    # Destroy the node explicitly
    image_node_base.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
