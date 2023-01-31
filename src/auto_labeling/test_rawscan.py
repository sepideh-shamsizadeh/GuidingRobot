#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan

images = []
pose_laser = []
def callback(poses):
    pose_laser.append(poses)


if __name__ == '__main__':
    rospy.loginfo('start matching')
    rospy.init_node('Matcher', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/scan_raw', LaserScan, callback, queue_size=2)
    rospy.spin()
    print(len(pose_laser))