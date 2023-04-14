#! usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

def generator1():
    num = 0
    while True:
        yield num
        num += 1


gen1 = generator1()

def generator2():
    num = 0
    while True:
        yield num
        num += 1


gen2 = generator2()

images = []
points = []

def scan_callback(scan_msg):
    # Do some processing on the laserscan data
    pass

def image_callback(image_msg):
    # Convert the image message to a OpenCV image
    images.append(image_msg)

    # Get the timestamp of the image message
    image_time = image_msg.header.stamp

    # Use the image timestamp to find the closest matching laserscan message
    scan_msg = rospy.wait_for_message('/scan', LaserScan, timeout=1.0)
    points.append(scan_msg)


    # Process the synchronized laserscan and image data
    scan_callback(scan_msg)


if __name__ == '__main__':
    rospy.init_node('image_laserscan_sync')
    rospy.Subscriber("/theta_camera/image_raw", Image, image_callback)
    rospy.spin()
    zzz = 0
    for img in images:
        i = next(gen1)
        if i == 1:
            cv_bridge = CvBridge()
            cv_image = cv_bridge.imgmsg_to_cv2(img, "bgr8")
            cv2.imwrite('../GuidingRobot/src/calib/' + str(i) + '.png', cv_image)

    for p in points:
        if zzz < 2:
            angles = np.linspace(0, 2 * np.pi, len(p.ranges))
            x = np.multiply(p.ranges, np.cos(angles))
            y = np.multiply(p.ranges, np.sin(angles))
            fig = plt.figure()
            plt.scatter(x, y, s=0.2)
            plt.show()
            zzz += 1

x


