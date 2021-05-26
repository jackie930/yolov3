from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def find_lower_lin(image):
    # 图像的阈值分割处理，即将图像处理成非黑即白的二值图像
    ret, image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # binary（黑白二值），ret代表阈值，80是低阈值，255是高阈值

    # 二值图像的反色处理，将图片像素取反
    height, width = image1.shape[:2]  # 返回图片大小
    image2 = image1.copy()
    for i in range(height):
        for j in range(width):
            image2[i, j] = (255 - image1[i, j])

    # 边缘提取，使用Canny函数
    image2_3 = cv2.Canny(image2, 80, 255)  # 设置80为低阈值，255为高阈值

    line = 1
    minLineLength = 1000
    maxLineGap = 150
    # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
    lines = cv2.HoughLinesP(image2_3, 1, np.pi / 180, 120, lines=line, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    lines1 = lines[:, 0, :]  # 降维处理
    # line 函数勾画直线
    # (x1,y1),(x2,y2)坐标位置
    # (0,255,0)设置BGR通道颜色
    # 2 是设置颜色粗浅度
    # print (lines1)
    res = []
    x1_ls = []
    x2_ls = []
    y1_ls = []
    y2_ls = []
    print ("image1.shape[1]", image1.shape[1])
    for i in lines1:
        x1, y1, x2, y2 = i
        if abs(y1 - y2) / y2 < 0.01:
            if x1 <= 1 and x2 >= image1.shape[1] - 3:
                x1_ls.append(x1)
                x2_ls.append(x2)
                y1_ls.append(y1)
                y2_ls.append(y2)

    print (res)
    idx = y1_ls.index(max(y1_ls, key=abs))
    print (idx)
    print (y1_ls[idx])

    x1, y1, x2, y2 = [x1_ls[idx], y1_ls[idx], x2_ls[idx], y2_ls[idx]]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # plt.imshow(image)
    # plt.show()
    return image


def infer(weights_path, image_folder, class_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet("./config/yolov3-custom.cfg", img_size=416).to(device)

    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.8, 0.4)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    result = {}
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        cv_img = cv2.imread(path)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, 416, cv_img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                print ("x1, y1, x2, y2: ", (x1, y1), (x2, y2))

                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_img, classes[int(cls_pred)], (x1, y1), font, 1, [225, 255, 0], 2)

        image_name = path.split('/')[-1]
        # save output images
        save_name = os.path.join('./output', image_name)
        cv2.imwrite(save_name, cv_img)  # save picture

    return result


if __name__ == "__main__":
    weights_path = './checkpoints/yolov3_ckpt_480.pth'
    image_folder = './data/custom/images'
    class_path = './data/custom/classes.names'
    infer(weights_path, image_folder, class_path)