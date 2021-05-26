# -*- coding: utf-8 -*-
"""
 @File    : yolo_label.py
 @Author  : Jackie
 @Description    :
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'valid']

classes = ["barcode"]  #

# sets generation
image_list = os.listdir('../data/custom/images')
label_list = os.listdir('../data/custom/labels')


def rename(file, keyword):
    ''' file: 文件路径    keyWord: 需要修改的文件中所包含的关键字 '''

    # os.chdir(file)
    items = os.listdir(file)
    # print(os.getcwd())
    for name in items:
        print(name)
        # 遍历所有文件
        if not os.path.isdir(name):
            if keyword in name:
                new_name = name.replace(keyword, '')
                os.renames(os.path.join(file, name), os.path.join(file, new_name))
        else:
            rename(file + '\\' + name, keyword)
            # os.chdir('...')


rename('../data/custom/labels', '.xml')

images = [i[:-4] for i in image_list]
print ("<<<<< length before", len(images))
xml_images = [i[:-4] for i in label_list]
print ("<<< images: ", images)
print ("<<< xml_images: ", xml_images)

images = [val for val in images if val in xml_images]
print ("<<<<< length after", len(images))

image_len = len(images)
num_train = image_len - int(image_len * 0.2)
num_test = int(image_len * 0.2)
print ("<<<< NUM TRAIN: ", num_train)
print ("<<<< NUM TEST: ", num_test)

print ("<<<< check if exisits")
if not os.path.exists('./data/custom'):
    os.makedirs('./data/custom')

train_file = open('../data/custom/train.txt', 'w')
test_file = open('../data/custom/valid.txt', 'w')
i = 0
for image_id in image_list:
    if i < num_train:
        # print (">>> images for train: ",image_id)
        train_file.write('%s\n' % ('data/custom/images/' + image_id))
    else:
        # print (">>> images for valid: ",image_id)
        test_file.write('%s\n' % ('data/custom/images/' + image_id))
    i = i + 1

train_file.close()
test_file.close()

