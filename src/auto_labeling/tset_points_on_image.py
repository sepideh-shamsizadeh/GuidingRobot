import math
import cv2
import numpy as np


def convert_robotF2imageF(tmpx, tmpy, side_info):
    H = side_info['H']
    fu = side_info['fu']
    fv = side_info['fv']
    u0 = side_info['u0']
    v0 = side_info['v0']
    Zc = H[4] * tmpx + H[5] * tmpy + H[8]
    u = ((fu * H[0] + u0 * H[4]) * tmpx + (fu * H[1] + u0 * H[5]) * tmpy + fu * H[6] + u0 * H[8]) / Zc
    v = ((fv * H[2] + v0 * H[4]) * tmpx + (fv * H[3] + v0 * H[5]) * tmpy + fv * H[7] + v0 * H[8]) / Zc
    return [u, v]




def draw_circle_bndBOX(u,v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)


f = '/home/sepid/workspace/Thesis/GuidingRobot/src/calib/ABback.txt'
A = []
B = []
xt = []
yt = []

back_info = {
    'H': np.array([-1.3272, -7.0239, -0.13689, 0.43081, 7.0104, -1.2212, -0.047192, 8.2577, -0.77688]),
    'fu': 250.001420127782,
    'fv': 253.955300723887,
    'u0': 239.731339559399,
    'v0': 246.917074981568
}

right_info = {
    'H': np.array([0.13646, -0.033852, -0.018656, 0.021548, 0.026631, 0.13902, -0.023934, 0.11006, -0.0037212]),
    'fu': 399373379354,
    'fv': 247.434371718165,
    'u0': 246.434570692999,
    'v0': 239.287976204900
}

left_info = {
    'H': np.array([0.15888, -0.036621, -0.021383, 0.025895, 0.030874, 0.16751, 0.035062, -0.16757, 0.002782]),
    'fu': 248.567135164434,
    'fv': 249.783014432268,
    'u0': 242.942149245269,
    'v0': 233.235264118894
}

front_info = {
    'H': np.array([-0.27263, -1.1756, 0.64677, -0.048135, 1.1741, -0.24661, -0.039707, -0.023353, -0.27371]),
    'fu': 239.720364104544,
    'fv': 242.389765646256,
    'u0': 237.571362200999,
    'v0': 245.039671395514
}
with open(f, 'r') as file:
    for line in file:
        line = line.strip()
        if line.endswith('.jpg'):
            filename = line
            print(filename)
            img = cv2.imread('/home/sepid/workspace/Thesis/GuidingRobot/src/auto_labeling/calib/scenes/back/'+filename)
        elif line.endswith(')'):
            A_value, B_value = line.strip('()\n').split(',')
            line = file.readline()
            xt_value, yt_value, _ = map(float, line.split(';'))
            u,v = convert_robotF2imageF(xt_value, yt_value, back_info)
            draw_circle_bndBOX(u,v,img)


