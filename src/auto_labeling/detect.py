import os
import sys
import time

import cv2
import numpy as np
import torch
from numpy import random

from src.yolov7.models.experimental import attempt_load
from src.yolov7.utils.datasets import letterbox
from src.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging, xyn2xy
from src.yolov7.utils.plots import plot_one_box
from src.yolov7.utils.torch_utils import select_device


def detect_person(img0):
    poses = []
    imgsz = 640
    conf_thres = 0.5
    iou_thres = 0.45

    weights = 'yolov7.pt'

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    here = os.path.dirname(os.path.abspath('yolov7'))
    sys.path.append(here)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    s = ''
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment='store_true')[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                if 'person' in label:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                    poses.append(xywh)

    # Stream results
    cv2.imshow('str(p)', img0)
    cv2.waitKey(0)  # 1 millisecond
    cv2.imwrite('/home/sepideh/Pictures/2.1.png', img0)

    return poses


if __name__ == '__main__':
    img0 = cv2.imread("/home/sepideh/Pictures/2.png")  # BGR
    cv2.imshow("image", img0)
    cv2.waitKey(0)
    p = detect_person(img0)
    print(p)
