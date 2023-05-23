#!/usr/bin/env python
import math
import csv

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import os
lsmsg=None

def counter():
    num = 0
    while True:
        yield num
        num += 1



def laser_scan2xy(msg):
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    print(angle_min, angle_increment)
    xy_points = []

    return xy_points


def scan_callbask(scan_msg):
    pass

def callback(imgs_msg):
    global image
    global lsmsg
    i = next(cnt)
    ls_msg = rospy.wait_for_message('/scan', LaserScan, timeout=1.0)
    lsmsg = ls_msg
    cvb = CvBridge()
    image = cvb.imgmsg_to_cv2(imgs_msg, 'bgr8')
    scan_callbask(ls_msg)
    laser_scan2xy(ls_msg)
    # cv2.imwrite('img' + str(i) + '.png', image)
    # with open('scan_data.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(ls_msg.ranges)


def draw_circle_bndBOX(u, v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    cnt = counter()
    rospy.loginfo('start test laser and camera')
    rospy.init_node('LaserCamera', anonymous=True, log_level=rospy.INFO)
    image_sub = rospy.Subscriber("/theta_camera/image_raw", Image, callback)
    file_path = '/home/sepid/workspace/Thesis/GuidingRobot/2023-05-10-15-30-23.bag'
    os.system('rosbag play ' + file_path + ' -d 1')

    # Shutdown the image subscriber after the first frame has been saved
    image_sub.unregister()







