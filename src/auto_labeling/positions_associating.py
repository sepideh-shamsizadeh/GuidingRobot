import yaml
import csv
import cv2
import ast
import math
import os
from PIL import Image
from math import pi, atan2, hypot, floor
from numpy import clip
import numpy as np

import detect_people

model = detect_people.load_model()


def counter():
    num = 0
    while True:
        yield num
        num += 1


def laser_scan2xy(msg):
    angle_min = -3.140000104904175
    angle_increment = 0.005799999926239252
    num_ranges = len(msg)
    xy_points = []
    left = []
    right = []
    front = []
    back = []
    for j in range(0, num_ranges):
        angle = angle_min + j * angle_increment
        r = float(msg[j])
        converted_angle = math.degrees(angle)
        if not math.isinf(r) and r > 0.1:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            x = round(x)
            y = round(y)
        if 45 <= converted_angle < 135:
            right.append((x, y))
        elif 135 <= converted_angle < 180 or -180 <= converted_angle < -135:
            front.append((x, y))
        elif -135 <= converted_angle < -45:
            left.append((x, y))
        elif -45 <= converted_angle < 45:
            back.append((x, y))
    return left, right, front, back


class CubeProjection:
    def __init__(self, imgIn, output_path):
        self.output_path = output_path
        self.imagin = imgIn
        self.sides = {
            'back': None,
            'left': None,
            'front': None,
            'right': None,
            'top': None,
            'bottom': None
        }

    def cube_projection(self):
        imgIn = self.imagin
        inSize = imgIn.size
        faceSize = int(inSize[0] / 4)

        FACE_NAMES = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
            4: 'top',
            5: 'bottom'
        }

        for face in range(6):
            imgOut = Image.new('RGB', (faceSize, faceSize), 'black')
            self.convertFace(imgIn, imgOut, face)
            if self.output_path != '':
                imgOut.save(self.output_path + FACE_NAMES[face] + '.jpg')
            else:
                self.sides[FACE_NAMES[face]] = imgOut

    def outImg2XYZ(self, i, j, faceIdx, faceSize):
        a = 2.0 * float(i) / faceSize
        b = 2.0 * float(j) / faceSize

        if faceIdx == 0:  # back
            (x, y, z) = (-1.0, 1.0 - a, 1.0 - b)
        elif faceIdx == 1:  # left
            (x, y, z) = (a - 1.0, -1.0, 1.0 - b)
        elif faceIdx == 2:  # front
            (x, y, z) = (1.0, a - 1.0, 1.0 - b)
        elif faceIdx == 3:  # right
            (x, y, z) = (1.0 - a, 1.0, 1.0 - b)
        elif faceIdx == 4:  # top
            (x, y, z) = (b - 1.0, a - 1.0, 1.0)
        elif faceIdx == 5:  # bottom
            (x, y, z) = (1.0 - b, a - 1.0, -1.0)
        return (x, y, z)

    def convertFace(self, imgin, imgout, faceIdx):
        inSize = imgin.size
        outsize = imgout.size
        inpix = imgin.load()
        outpix = imgout.load()
        facesize = outsize[0]

        for xout in range(facesize):
            for yout in range(facesize):
                (x, y, z) = self.outImg2XYZ(xout, yout, faceIdx, facesize)
                theta = atan2(y, x)  # range -pi to pi
                r = hypot(x, y)
                phi = atan2(z, r)  # range -pi/2 to pi/2

                # source img coords
                uf = 0.5 * inSize[0] * (theta + pi) / pi
                vf = 0.5 * inSize[0] * (pi / 2 - phi) / pi

                # Use bilinear interpolation between the four surrounding pixels
                ui = floor(uf)  # coord of pixel to bottom left
                vi = floor(vf)
                u2 = ui + 1  # coords of pixel to top right
                v2 = vi + 1
                mu = uf - ui  # fraction of way across pixel
                nu = vf - vi

                # Pixel values of four corners
                A = inpix[int(ui % inSize[0]), int(clip(vi, 0, inSize[1] - 1))]
                B = inpix[int(u2 % inSize[0]), int(clip(vi, 0, inSize[1] - 1))]
                C = inpix[int(ui % inSize[0]), int(clip(v2, 0, inSize[1] - 1))]
                D = inpix[int(u2 % inSize[0]), int(clip(v2, 0, inSize[1] - 1))]

                # interpolate
                (r, g, b) = (
                    A[0] * (1 - mu) * (1 - nu) + B[0] * (mu) * (1 - nu) + C[0] * (1 - mu) * nu + D[0] * mu * nu,
                    A[1] * (1 - mu) * (1 - nu) + B[1] * (mu) * (1 - nu) + C[1] * (1 - mu) * nu + D[1] * mu * nu,
                    A[2] * (1 - mu) * (1 - nu) + B[2] * (mu) * (1 - nu) + C[2] * (1 - mu) * nu + D[2] * mu * nu)

                outpix[xout, yout] = (int(round(r)), int(round(g)), int(round(b)))


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


def check_intersection(d_bound, point):
    if d_bound[0] < point[0] < d_bound[2]:
        if d_bound[1] < point[1] < d_bound[3]:
            return True
    return False


def check_points(x_y, p_uv, person):
    avgx = person[0] + abs(person[0] - person[2]) / 2
    avgy = person[1] + abs(person[1] - person[3]) / 2
    # print('avg', avgx, avgy)
    inside = []
    min_index = 0
    for p in p_uv:
        # print(p)
        inside.append((abs(avgx - p[0]), abs(avgy - p[1])))
    min_index = min(range(len(inside)), key=lambda i: inside[i])
    # print('in', inside)
    # print('min', min_index)
    return x_y[min_index]


def selected_point(side_xy, side_info, face, detected):
    # print(face)
    XY_people = []
    # print(detected)
    for person in detected:
        p = []
        x_y = []
        x = 0
        y = 0
        for xy in side_xy:
            u, v = convert_robotF2imageF(xy[0], xy[1], side_info)
            if face == 'back':
                v -= 45
            if face == 'left':
                u += 30
                v -= 20
            # print('u,v', u, v)
            # draw_circle_bndBOX(u, v, cv_image)
            # print('(u, v)', (u, v))
            # print('x,y', (xy[0], xy[1]))
            if check_intersection(person, (u, v)):
                p.append((u, v))
                x_y.append((xy[0], xy[1]))

        # print('len', len(p))
        if len(p) > 1:
            x, y = check_points(x_y, p, person)
        elif len(p) == 1:
            x, y = x_y[0]
        # print('x_y', x_y)
        # print('x,y', x, y)
        XY_people.append((x, y))

    return XY_people


def draw_circle_bndBOX(u, v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    FACE_NAMES = ['back', 'front', 'left', 'right']
    counter_gen = counter()

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
    scan = []
    with open('/home/sepid/workspace/Thesis/GuidingRobot/data/scan.csv', 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        for row in reader:
            image_id = int(row[0])  # Extract the image ID from the first column
            ranges = [float(value) for value in row[1:]]  # Extract th
            scan.append(ranges)

    dr_spaam = []
    with open('/home/sepid/workspace/Thesis/GuidingRobot/data/drspaam4_data.csv', 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        for row in reader:
            dr_spaam.append(row)

    data = {}
    # for i in range(36, int(len(scan)/2)):
    for i in range(len(dr_spaam)):

        path = '/home/sepid/workspace/Thesis/GuidingRobot/data/image_' + str(i) + '.jpg'
        print(path)
        if os.path.exists(path):
            img = cv2.imread(path)
#             # print()
            left, right, front, back = laser_scan2xy(scan[i])
            back_xy = []
            left_xy = []
            right_xy = []
            front_xy = []
            for d in dr_spaam[i]:
                dr_value = tuple_of_floats = ast.literal_eval(d)
                dd = (round(dr_value[0]), round(dr_value[1]))
                if dd in back:
                    back_xy.append((dr_value[0], dr_value[1]))
                elif dd in front:
                    front_xy.append((dr_value[0], dr_value[1]))
                elif dd in right:
                    right_xy.append((dr_value[0], dr_value[1]))
                elif dd in left:
                    left_xy.append((dr_value[0], dr_value[1]))
            # cv2.imshow('img1', img)
            # cv2.waitKey(0)
            img = Image.fromarray(img)
            sides = CubeProjection(img, '')
            sides.cube_projection()
            people = []
            print(dr_spaam[i])
            for face, side_img in sides.sides.items():
                if face in FACE_NAMES:
                    cv_image = np.array(side_img)
                    detected = detect_people.detect_person(cv_image, model)
#                     # print(detected)

                    if face == 'back':
                        XY = selected_point(back_xy, back_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'front':
                        XY = selected_point(front_xy, front_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'right':
                        XY = selected_point(right_xy, right_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'left':
                        XY = selected_point(left_xy, left_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

            data = []
            # print(people)
            people = list(dict.fromkeys(people))
            if len(people) > 0:
                for k, p in enumerate(people):
                    if abs(p[0]) < 10 and abs(p[1]) < 10:
                        position = {'x': p[0], 'y': p[1]}
                        pp = {'id' + str(k): position}
                        data.append(pp)
                yaml_data = {'frame ' + str(next(counter_gen)): data}
                output_file = 'output1.yaml'

                # Open the file in write mode
                with open(output_file, 'a') as file:
                    # Write the YAML data to the file
                    yaml.dump(yaml_data, file)
