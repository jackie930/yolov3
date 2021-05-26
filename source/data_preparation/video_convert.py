# -*- coding: utf-8 -*-
"""
 @File    : convert_video.py
 @Author  : Jackie
 @Description    :
"""

import cv2
import os
import sys
from shutil import copyfile

def convert_video(video_path,dst_folder,EXTRACT_FREQUENCY):
    """video convert to frames"""
    if not os.path.exists(video_path):
        print("can not find the video")
        exit(1)

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    count = 1
    index = 0

    video = cv2.VideoCapture(video_path)

    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(index - 1))

def preprocess(imgfolder,targetfolder):
    image_list = os.listdir(imgfolder)
    print ('total number:', len(image_list))

    if not os.path.isdir(targetfolder):
        os.makedirs(targetfolder)

    for i in range(len(image_list)):
        #print(image_list[i])
        # 遍历所有文件
        source = os.path.join(imgfolder, image_list[i])
        target = os.path.join(targetfolder, str(i)+'.jpg')

        # adding exception handling
        try:
            copyfile(source, target)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

def preprocess_imgs(imgfolder,targetfolder):
    """if there are target imgs also, move the the same folder as video images,make sure to rename the imgs"""
    return

if __name__ == "__main__":
    # 全局变量
    VIDEO_PATH = '/Users/liujunyi/Desktop/spottag/summit-training/road/pics/1.mov' # 视频地址
    EXTRACT_FOLDER = './ims/' # 存放帧图片的位置
    EXTRACT_FREQUENCY = 10 # 帧提取频率
    convert_video(VIDEO_PATH, EXTRACT_FOLDER, EXTRACT_FREQUENCY)
    print ('success!')
    #preprocess_imgs(imgfolder,EXTRACT_FOLDER)
    preprocess(EXTRACT_FOLDER, './imgs_refined/')
    print ('success refined!')

