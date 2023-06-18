import rospy
import cv2
import csv
import numpy as np
from sensor_msgs.msg import Image, LaserScan
import time

rospy.init_node('sync_image_scan', anonymous=True)

def image_callback(msg):
    scan_filename = f"/home/sepid/workspace/Thesis/GuidingRobot/data1/image.csv"
    with open(scan_filename, 'a') as csvfile:
        image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        writer = csv.writer(csvfile)
        writer.writerow([msg.header.stamp] + list(image_data))

def scan_callback(msg):
    scan_filename = f"/home/sepid/workspace/Thesis/GuidingRobot/data1/scan.csv"
    with open(scan_filename, 'a') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow([msg.header.stamp] + list(msg.ranges))

rospy.Subscriber('/theta_camera/image_raw', Image, image_callback)
rospy.Subscriber('/scan', LaserScan, scan_callback)

previous_time = time.time()

rospy.spin()

