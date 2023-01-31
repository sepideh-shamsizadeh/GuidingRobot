import cv2
import numpy as np


def break_videos2frames(vid, fld):
    print(fld)
    cap = cv2.VideoCapture(vid)
    while not cap.isOpened():
        cap = cv2.VideoCapture(vid)
        cv2.waitKey(1000)
        print("Wait for the header")
    i = 0
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            if i % 10 == 0:
                cv2.imwrite(fld + str(i) + '.jpg', frame)
                if 'front' in fld:
                    frame = np.concatenate((frame[:, 1440:1920, :], frame[:, 0:480, :]), axis=1)
                    fld1 = 'checkerboard_images/front_separate/'
                    cv2.imshow('video', frame)
                    cv2.waitKey(1)
                    cv2.imwrite(fld1 + str(i) + '.jpg', frame)
                if 'rear' in fld:
                    frame = frame[:, 480:1440, :]
                    fld1 = 'checkerboard_images/rear_separate/'
                    cv2.imshow('video', frame)
                    cv2.waitKey(1)
                    cv2.imwrite(fld1 + str(i) + '.jpg', frame)
            i += 1
        else:
            break


if __name__ == '__main__':
    folders = ['front', 'rear', 'left', 'right']
    videos = ["front_lens.mp4", "rear_lens.mp4", "left.mp4", "right.mp4"]
    for v, f in zip(videos, folders):
        f = 'checkerboard_images/' + f + '/'
        break_videos2frames(v, f)
