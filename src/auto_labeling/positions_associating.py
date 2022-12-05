import math
import cv2


def get_laser_positions():
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
    return la_position


def get_image_postions():
    im_p = []
    with open('../../src/yolov7/im_pose.txt') as f:
        contents = f.readlines()
        for i, line in enumerate(contents):
            if ']' in line:
                x = line.split(']')[0]
                c = x.split('[')[1].split(',')
                b = []
                for z in c:
                    b.append(int(z))
                im_p.append(b)
            else:
                im_p.append([])
    return im_p


def convert_robotF2imageF(pose):
    theta = math.atan2(pose[1], pose[0])
    p = (pose[1]) / math.sin(theta)
    t = math.degrees(theta)
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
        cv2.circle(img, (960, 480), 10, (255, 0, 0), 3)
    if len(im_p) == 4:
        cv2.rectangle(img, (im_p[0], im_p[1]), (im_p[2], im_p[3]), (0, 255, 0), 1)
    cv2.imshow("imag1e", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    source = "../../src/images/"
    la_position = get_laser_positions()
    im_p = get_image_postions()
    for i in range(50, 60):
        img0 = cv2.imread(source + str(i) + '.png')  # BGR
        for j in range(i - 1, i + 10):
            draw_circle_bndBOX(la_position[j], im_p[i], img0)
