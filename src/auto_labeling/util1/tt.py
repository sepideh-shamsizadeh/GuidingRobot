import csv
import math
import matplotlib.pyplot as plt
import ast
import os

angle_min = -3.140000104904175
angle_increment = 0.005799999926239252
downsample_factor = 3  # Adjust the downsample factor as needed

with open('/home/sepid/workspace/Thesis/GuidingRobot/data1/scan.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    for i, row in enumerate(reader1):
        xx = []
        yy = []
        path = '/home/sepid/workspace/Thesis/GuidingRobot/data1/image_' + str(i) + '.jpg'
        print(path)
        if os.path.exists(path):
            # Read each row of the CSV file
            ranges = [float(value) for value in row]
            num_ranges = len(ranges)
            for j in range(num_ranges):
                angle = angle_min + j * angle_increment
                r = float(ranges[j])

                if not math.isinf(r) and r > 0.1:
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    if j % downsample_factor == 0:  # Downsample the points
                        xx.append(x)
                        yy.append(y)
            # Plot the laser scan data
            fig, ax = plt.subplots()
            ax.plot(xx, yy, 'bo', markersize=2)
            plt.show()
            plt.pause(0.01)