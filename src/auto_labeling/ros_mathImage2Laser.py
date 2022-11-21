import rospy
from sensor_ros.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import detect
from geometry_msgs.msg import PoseArray, Pose


pose_image = []
pose_laser = []


def callback_image(data):
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr('Converting Image Error.' + str(e))
        return
    pose_image.append(detect.detect_person(cv_image))

def callback(poses):
    pose = Pose()


if __name__ == '__main__':
    rospy.loginfo('start matching')
    rospy.init_node('Matcher', anonymous=True, log_level=rospy.INFO)
    rospy.Subscriber('/theta_camera/image_raw', Image, callback_image, queue_size=2, buff_size=2**24)
    rospy.Subscriber('/dr_spamm_detection', PoseArray, callback, queue_size=2)
