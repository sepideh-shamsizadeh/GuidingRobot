import cv2
import os
import numpy as np
img_array = []

folder = '../../../src/images'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
for i in range(len(img_array)):
    out.write(img_array[i])
    out.release()