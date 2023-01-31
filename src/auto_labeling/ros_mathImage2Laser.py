import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from detect import detect_person, load_model
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image


images = []
pose_laser = []
model = load_model()


def callback_image(data):
    images.append(data)


def callback(poses):
    pose_laser.append(poses.poses)


if __name__ == '__main__':
    rospy.loginfo('start matching')
    rospy.init_node('Matcher', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/theta_camera/image_raw', Image, callback_image, queue_size=2)
    rospy.Subscriber('/dr_spaam_detections', PoseArray, callback, queue_size=2)
    rospy.spin()
    print(len(pose_laser), len(images))
    for i, p in enumerate(pose_laser):
        with open('la_pose.txt', 'a') as f:
            f.write(" ".join(str(item) for item in p))
            f.write('*************************'+str(i)+'************************************')
            f.write('\n')
    pose_laser = []
    for i, data in enumerate(images):
        cv_bridge = CvBridge()
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imwrite('../images/' + str(i) + '.png', cv_image)
        detected = detect_person(cv_image, model)
        with open('im_pose.txt', 'a') as f:
            f.write(" ".join(str(item) for item in detected))
            f.write('*************************' + str(i) + '************************************')
            f.write('\n')
        images[i] = []
