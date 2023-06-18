import rospy
from sensor_msgs.msg import LaserScan, Image
import csv
import cv2
import numpy as np

scan_buffer = None  # Buffer for storing the latest received image message
scan_csv_file = "/home/sepid/workspace/Thesis/GuidingRobot/data1/scan.csv"  # File name for saving scan data
inds = 0
def counter():
    num = 0
    while True:
        yield num
        num += 1
def image_callback(image_msg):

    save_image(image_msg)

def scan_callback(scan_msg):
    global inds
    inds =next(ind)
    print(inds)
    save_scan(scan_msg)

def save_scan(scan_msg):
    if inds%5 == 0:
        with open(scan_csv_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([int(inds/5)] + list(scan_msg.ranges))

def save_image(image_msg):
    if inds % 5 == 0:
        image_file = '/home/sepid/workspace/Thesis/GuidingRobot/data1/image_'+str(int(inds/5))+'.jpg'
        try:
            image_data = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width, -1))
            if image_data is None:
                rospy.logerr('Failed to decode the image')
                return
            cv2.imwrite(image_file, image_data)
        except Exception as e:
            rospy.logerr('An error occurred while saving the image: %s', str(e))


rospy.init_node('scan_image_synchronizer')
ind = counter()

scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback)
image_sub = rospy.Subscriber('/theta_camera/image_raw', Image, image_callback)

rospy.spin()
