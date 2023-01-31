# !/usr/bin/env python
import glob
import cv2
import numpy as np
import os

DIM=(960, 960)
K=np.array([[325.53366320616897, 0.0, 491.189167792393], [0.0, 324.4750809403645, 454.2061993406636], [0.0, 0.0, 1.0]])
D=np.array([[-0.10403156166663023], [0.23333507260530398], [-0.1941365883840621], [0.05448534910345762]])


def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite(img_path, undistorted_img)
    cv2.waitKey(0)
    cv2.imshow("orginal", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images = glob.glob('checkerboard_images/separates/*.jpg')
    for fname in images:
        undistort(fname)

    # mtx =np.array( [[950.9105427,    0.,         963.61323739],
    #  [0., 948.56426089, 481.42879285],
    # [0.,0.,1.]])
    #
    # dist= np.array([[-0.0618459, -0.04816405,  0.0029544, -0.01144255 , 0.02641483]])





    # images = glob.glob('checkerboard_images/separates/*.jpg')
    # for i in images:
    #     img = cv2.imread(i)
    #     h, w = img.shape[:2]
    #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 0, (h, w))
    #     x, y, w1, h1 = roi
    #     yh1 = y + h1
    #     xw1 = x + w1
    #     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #     cv2.imshow("undistorted", dst)
    #     cv2.imwrite(i+'1.jpg' ,dst)
    #     cv2.waitKey(0)
    #     cv2.imshow("orginal", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

