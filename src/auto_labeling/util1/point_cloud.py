import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


def callback(point_cloud):
    # Convert the point cloud message to a generator of points
    points = pc2.read_points(point_cloud)

    # Iterate over each point and print its contents
    for point in points:
        print("point={}".format(point))

        # Extract x, y, and z coordinates if there are exactly 3 values in the point
        if len(point) == 3:
            x, y, z = point
            print("x={}, y={}, z={}".format(x, y, z))


rospy.init_node('point_cloud_subscriber')
rospy.Subscriber('/merged_cloud', PointCloud2, callback)
rospy.spin()