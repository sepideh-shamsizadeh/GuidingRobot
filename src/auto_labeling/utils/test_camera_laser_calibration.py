#!/usr/bin/env python
import math

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np


def counter():
    num = 0
    while True:
        yield num
        num += 1


def laser_scan2xy(msg):
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    ranges = msg.ranges
    num_ranges = len(ranges)
    xy_points = []
    for j in range(0, num_ranges):
        angle = angle_min + j * angle_increment
        r = ranges[j]
        if not math.isinf(r) and r > 0.1:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            xy_points.append((x, y))

    return xy_points


def scan_callbask(scan_msg):
    pass



def callback(imgs_msg):
    global image
    global lsmsg
    i = next(cnt)
    ls_msg = rospy.wait_for_message('/scan', LaserScan, timeout=1.0)
    if i == 10:
        lsmsg = ls_msg
        cvb = CvBridge()
        image = cvb.imgmsg_to_cv2(imgs_msg, 'bgr8')
    scan_callbask(ls_msg)


if __name__ == '__main__':
    cnt = counter()
    rospy.loginfo('start test laser and camera')
    rospy.init_node('LaserCamera', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber("/theta_camera/image_raw", Image, callback)
    rospy.spin()
    xy = laser_scan2xy(lsmsg)
    H = [-3.10066197072063e-07, -3.17251174466163e-08, -7.31843145153427e-07,
         7.9749e-08, 1.81805732359934e-08, -3.85206729780813e-07,
         2.44775143318384e-07, 1.59694345877475e-08, 5.71521553290399e-07]
    fu = 767.413
    fv = 884.984
    u0 = 962.699
    v0 = 69.412
    for point in xy:
        tmpx = point[0]
        tmpy = point[1]
        Zc = H[4] * tmpx + H[5] * tmpy + H[8]
        u = ((fu * H[0] + u0 * H[4]) * tmpx + (fu * H[1] + u0 * H[5]) * tmpy + fu * H[6] + u0 * H[8]) / Zc
        v = ((fv * H[2] + v0 * H[4]) * tmpx + (fv * H[3] + v0 * H[5]) * tmpy + fv * H[7] + v0 * H[8]) / Zc
        print(u, v)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)



