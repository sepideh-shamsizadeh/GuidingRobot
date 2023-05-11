#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from detect import detect_person, load_model
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image


model = load_model()


def callback_image(data):
    i= next(gen1)
    cv_bridge = CvBridge()
    cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imwrite('../images/' + str(i) + '.png', cv_image)
    detected = detect_person(cv_image, model)
    with open('im_pose.txt', 'a') as f:
        f.write(" ".join(str(item) for item in detected))
        f.write('*********************'+str(i)+'****************************************')
        f.write('\n')


def callback(poses):
    i = next(gen2)
    with open('la_pose.txt', 'a') as f:
        f.write(" ".join(str(item) for item in poses.poses))
        f.write('*****************************'+str(i)+'********************************')
        f.write('\n')


def generator1():
    value = 0
    # produce the current value of the counter
    yield value

    # increment the counter
    value += 1


def generator2():
    value = 0
    # produce the current value of the counter
    yield value

    # increment the counter
    value += 1


if __name__ == '__main__':
    gen1 = generator1()
    gen2 = generator2()
    rospy.loginfo('start matching')
    rospy.init_node('Matcher', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/theta_camera/image_raw', Image, callback_image, queue_size=2)
    rospy.Subscriber('/dr_spaam_detections', PoseArray, callback, queue_size=2)
    rospy.spin()

