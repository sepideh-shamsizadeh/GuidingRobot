#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

pose_laser = []
bridge = CvBridge()

images =[]
def callback_image(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imshow('image', cv_image)
    cv2.waitKey(1)


if __name__ == '__main__':
    rospy.loginfo('start matching')
    rospy.init_node('test_camera', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/theta_camera/image_raw', Image, callback_image, queue_size=5)
    rospy.spin()





