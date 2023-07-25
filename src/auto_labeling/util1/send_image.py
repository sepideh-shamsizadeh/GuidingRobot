#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header


import cv2

from abc import ABCMeta, abstractmethod  # Abstract class.

cv_bridge = CvBridge()


def to_ros_image(cv2_uint8_image, img_format="bgr"):
    # -- Check input.
    shape = cv2_uint8_image.shape  # (row, col, depth=3)
    print(shape)
    assert(len(shape) == 3 and shape[2] == 3)

    # -- Convert image to bgr format.
    if img_format == "rgb":  # If rgb, convert to bgr.
        print('rgb')
        bgr_image = cv2.cvtColor(cv2_uint8_image, cv2.COLOR_RGB2BGR)
    elif img_format == "bgr":
        print('****')
        bgr_image = cv2_uint8_image
    else:
        raise RuntimeError("Wrong image format: " + img_format)

    # -- Convert to ROS format.
    ros_image = cv_bridge.cv2_to_imgmsg(img, "bgr8")
    return ros_image


if __name__ == '__main__':
    node_name = "publish_images"
    rospy.init_node(node_name)
    img = cv2.imread("outputImage.jpg")
    ros_img = to_ros_image(img)
    pub = rospy.Publisher('/theta_camera/image_raw', Image, queue_size=5)
    while True:
        pub.publish(ros_img)
        rospy.sleep(0.1)
    rospy.spin()

