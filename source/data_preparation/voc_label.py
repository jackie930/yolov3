# -*- coding: utf-8 -*-
"""
 @File    : voc_label.py
 @Author  : Jackie
 @Description    :
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'valid']

classes = ["barcode"]#我


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('data/Annotations/%s.xml' % (image_id))
    out_file = open('data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        print (xmlbox)
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#sets generation
image_list = os.listdir('./data/images')
xml_list = os.listdir('./data/Annotations')

images = [i[:-4] for i in image_list]
print ("<<<<< length before", len(images))
xml_images = [i[:-4] for i in xml_list]

images = [val for val in images if val in xml_images]
print ("<<<<< length after", len(images))

image_len = len(images)
num_train = image_len - int(image_len*0.2)
num_test = int(image_len*0.2)
print ("<<<< NUM TRAIN: ", num_train)
print ("<<<< NUM TEST: ", num_test)

print ("<<<< check if exisits")
if not os.path.exists('./data/ImageSets'):
    os.makedirs('./data/ImageSets')

print ("files: ", os.listdir('./data'))

train_file = open('./data/ImageSets/train.txt', 'w')
test_file = open('./data/ImageSets/valid.txt', 'w')
i = 0
for image_id in images:
    if i<num_train:
        #print (">>> images for train: ",image_id)
        train_file.write('%s\n' % (image_id))
    else:
        #print (">>> images for valid: ",image_id)
        test_file.write('%s\n' % (image_id))
    i = i+1

train_file.close()
test_file.close()

class_file = open('./data/classes.names', 'w')
for i in classes:
    class_file.write('%s\n' % (i))


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    if not os.path.exists('data/ImageSets/'):
        os.makedirs('data/ImageSets/')

    image_ids = open('./data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('./data/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data/custom/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
