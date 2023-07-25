import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np

# Open the CSV file
with open('calib/scan_rear.csv', newline='') as csvfile:
    # Read the CSV data into a list of rows
    reader = csv.reader(csvfile)

    # Skip the header row
    next(reader)

    # Iterate over each row and plot the laser scan data
    for row in reader:
        # Extract the range measurements from the 'ranges' column

        # Compute the number of range measurements and the angle increment
        num_ranges = len(ranges)
        angle_increment = 2 * math.pi / num_ranges

        # Create a list of angles
        angles = [i * angle_increment for i in range(num_ranges)]
        print(angles)
        # Plot the laser scan data
        fig, ax = plt.subplots()
        ax.plot(angles, ranges)
        ax.set_ylim(0, 10)
        plt.show()
