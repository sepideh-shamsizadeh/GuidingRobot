import math
import cv2


def convert_robotF2imageF(pose):
    theta = math.atan2(pose['y'], pose['x'])
    p = pose['y'] / math.sin(theta)
    t = math.degrees(theta)
    pp = math.degrees(p)
    x = (180 - t) * (1920 / 360)
    y = (180 - pp) * (920 / 360)
    print(x, y)


def check_positions(image_positions, laser_positions):
    poses = []
    for x, y in laser_positions:
        for x1, y1, x2, y2 in image_positions:
            if x1 <= x <= x2:
                if y1 <= y < y2:
                    poses.append([x, y])
    return poses


def draw_circle_bndBOX(poses, im_p, img):
    print(poses)
    for pose in poses:
        cv2.circle(img, (pose[0], pose[1]), 10, (255, 0, 0), 1)
    for p in im_p:
        print(p)
        print((int(p[0]), int(p[1])), (int(p[2]), int(p[3])))
        print(img.shape)
        cv2.rectangle(img, (int(p[0]), int(p[1])), (p[2], p[3]), (0, 255, 0), 1)
    cv2.imshow("imag1e", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    im_p = [[1365, 481, 1456, 718]]
    l_p = [[434, 343], [455, 330], [912, 294], [902, 295], [225, 320], [8, 246], [79, 360], [938, 252]]
    poses = check_positions(im_p, l_p)
    source = "../../src/images/"
    img0 = cv2.imread(source + '207.png')  # BGR
    cv2.imshow("image", img0)
    cv2.waitKey(0)
    draw_circle_bndBOX(poses, im_p, img0)
    pose = {
        'x': 1.42,
        'y': 0.09
    }
    convert_robotF2imageF(pose)
