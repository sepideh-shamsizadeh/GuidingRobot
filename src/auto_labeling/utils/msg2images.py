#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


def counter():
    num = 0
    while True:
        yield num
        num += 1


def callback(img):
    global last_saved_time
    cvb = CvBridge()
    cvi = cvb.imgmsg_to_cv2(img, 'bgr8')
    current_time = rospy.Time.now()
    if (current_time - last_saved_time).to_sec() > 0.1:
        cv2.imwrite('src/calib/checkerboard/' + str(next(cnt)) + '.png', cvi)
        last_saved_time = current_time


if __name__ == '__main__':
    cnt = counter()
    rospy.loginfo('start')
    rospy.init_node('Msg2Image', anonymous=True, log_level=rospy.INFO)
    last_saved_time = rospy.Time.now()
    rospy.Subscriber('/theta_camera/image_raw', Image, callback, queue_size=2)
    rospy.spin()
