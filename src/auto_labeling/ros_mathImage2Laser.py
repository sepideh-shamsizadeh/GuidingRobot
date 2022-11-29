import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from detect import detect_person, load_model
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image


pose_image = []
pose_laser = []


def callback_image(data):
    pose_image.append(data)


def callback(poses):
    with open('la_pose.txt', 'a') as f:
        f.write(" ".join(str(item) for item in poses.poses))
        f.write('\n')
        pose_laser.append(poses.poses)


if __name__ == '__main__':
    rospy.loginfo('start matching')
    rospy.init_node('Matcher', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/theta_camera/image_raw', Image, callback_image, queue_size=2)
    rospy.Subscriber('/dr_spaam_detections', PoseArray, callback, queue_size=2)
    rospy.spin()
    print(len(pose_laser), len(pose_image))

