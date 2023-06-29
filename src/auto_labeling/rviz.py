import rospy
from sensor_msgs.msg import LaserScan
import csv
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Initialize ROS node
rospy.init_node('range_publisher', anonymous=True)

# Create a publisher to publish the LaserScan message
scan_publisher = rospy.Publisher('/scan', LaserScan, queue_size=10)

# Create a LaserScan message
scan_msg = LaserScan()
# Create a publisher for the '/theta_camera/image_raw' topic
image_pub = rospy.Publisher('/theta_camera/image_raw', Image, queue_size=10)

# Create a CvBridge object
bridge = CvBridge()
# Set the necessary fields of the LaserScan message
scan_msg.header.frame_id = "base_link"  # Set the appropriate frame ID
scan_msg.angle_min = -3.140000104904175  # Set the minimum angle
scan_msg.angle_max = 3.140000104904175  # Set the maximum angle
scan_msg.angle_increment = 0.005799999926239252  # Set the angle increment
scan_msg.range_min = 0.44999998807907104  # Set the minimum range value
scan_msg.range_max = 25.0  # Set the maximum range value

angle_min = -3.140000104904175
angle_increment = 0.005799999926239252

downsample_factor = 3  # Adjust the downsample factor as needed

with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/scan.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    for i, row in enumerate(reader1):
        xx = []
        yy = []
        path = '/home/sepid/workspace/Thesis/GuidingRobot/data2/image_' + str(i) + '.jpg'
        if os.path.exists(path):
            ranges = [float(value) for value in row]  # Read the ranges from the CSV file
            scan_msg.ranges = ranges  # Set the range values in the LaserScan message

            # Publish the LaserScan message
            scan_publisher.publish(scan_msg)
            cv2_image = cv2.imread(path)
            ros_image = bridge.cv2_to_imgmsg(cv2_image, encoding="bgr8")

            # Publish the ROS message
            image_pub.publish(ros_image)

            # Wait for keyboard input before publishing the next message
            input("Press Enter to publish the next scan message...")
            print(path)

