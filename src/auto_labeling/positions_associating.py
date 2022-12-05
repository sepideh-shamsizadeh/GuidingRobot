import math
import cv2


def convert_robotF2imageF(pose):
    theta = math.atan2(pose[1], pose[0])
    p = (pose[1]) / math.sin(theta)
    t = -math.degrees(theta)
    pp = math.degrees(p)
    print(pp, t)
    x = (180 - t) * (1920 / 360)
    y = (90 - pp) * (960 / 180)
    return [x, y]


def check_positions(image_positions, laser_positions):
    poses = []
    for x, y in laser_positions:
        for x1, y1, x2, y2 in image_positions:
            if x1 <= x <= x2:
                if y1 <= y < y2:
                    poses.append([x, y])
    return poses


def draw_circle_bndBOX(poses, im_p, img):
    for pose in poses:
        cv2.circle(img, (int(pose[0]), int(pose[1])), 10, (0, 0, 255), 3)
        cv2.circle(img, (960,480), 10, (0, 210, 255), 3)
    for p in im_p:
        cv2.rectangle(img, (int(p[0]), int(p[1])), (p[2], p[3]), (0, 255, 0), 1)
    cv2.imshow("imag1e", img)
    cv2.waitKey(1)


if __name__ == '__main__':
    im_p = [[1171, 501, 1256, 763]]
    source = "../../src/images/"
    positions = []
    frame_pose = []
    with open('../../src/yolov7/la_pose.txt') as f:
        contents = f.readlines()
        for i, line in enumerate(contents):
            if 'position' in line:
                frame_pose.append([float(contents[i + 1].split(':')[1].split(' ')[1]),
                                   float(contents[i + 2].split(':')[1].split(' ')[1])])
            if '********' in line:
                positions.append(frame_pose)
                frame_pose = []

    la_position = []
    for pose in positions:
        l = []
        for la_p in pose:
            l.append(convert_robotF2imageF(la_p))
        la_position.append(l)

    for i in range(50, 1500):
        img0 = cv2.imread(source + str(i)+'.png')  # BGR
        for j in range(i-1, i+187):
            draw_circle_bndBOX(la_position[j], im_p, img0)

