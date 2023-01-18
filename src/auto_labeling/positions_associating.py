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
    im = []
    with open('../../src/yolov7/im_pose.txt') as f:
        contents = f.readlines()
        for i, line in enumerate(contents):
            if ']' in line:
                x = line.split(']')
                for xx in x:
                    if '[' in xx:
                        c = xx.split('[')[1].split(',')
                        b = []
                        for z in c:
                            b.append(int(z))
                        im.append(b)
            im_p.append(im)
            im = []
    return im_p


def convert_robotF2imageF(pose):
    O0x = int(1920/2)
    O0y = int(960/2)
    O1x = 1360
    O1y = 770
    theta = math.radians(0)
    pix_x = pose[0]*200
    pix_y = pose[1]*200
    x = O1x + pix_x*math.cos(theta) - pix_y*math.sin(theta)
    y = O1y - (pix_x*math.sin(theta) + pix_y*math.cos(theta))

    return [x, y]


def check_positions(image_positions, laser_positions):
    poses = []
    for x, y in laser_positions:
        for x1, y1, x2, y2 in image_positions:
            if x1 <= x <= x2:
                if y1 <= y < y2:
                    poses.append([x, y])
    return poses


def draw_circle_bndBOX(poses, im_p, img, color=(255, 0, 0)):
    for pose in poses:
        cv2.circle(img, (int(pose[0]), int(pose[1])), 10, (0, 0, 255), 3)
        cv2.circle(img, (1360, 770), 10, color, 3)
        cv2.circle(img, (960, 480), 10, (22, 180, 77), 3)
        cv2.line(img, (960, 0), (960, 960), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(img, (0, 480), (1920, 480), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(img, (1160, 770), (1360, 770), (255, 0, 0), thickness=3, lineType=8)
        cv2.line(img, (1360, 770), (1360, 970), (255, 0, 0), thickness=3, lineType=8)
    if len(im_p) == 4:
        cv2.rectangle(img, (im_p[0], im_p[1]), (im_p[2], im_p[3]), (0, 255, 0), 1)
    cv2.imshow("imag1e", img)
    cv2.waitKey(1)


if __name__ == '__main__':
    source = "../../src/images/"
    p = convert_robotF2imageF([0, 1])
    print(p)
    la_position = get_laser_positions()
    im_p = get_image_postions()
    img0 = cv2.imread(source + str(0) + '.png')  # BGR

    for i in range(0, 1047):
        if i < 1000:
            j = i
            img0 = cv2.imread(source + str(i) + '.png')  # BGR
        for m in range(len(im_p[j])):
            draw_circle_bndBOX(la_position[i], im_p[j][m], img0)
