# -*- coding: utf-8 -*-
"""
 @File    : yolo_label.py
 @Author  : Jackie
 @Description    :
"""

import json
import os
from shutil import copyfile
from sys import exit

sets = ['train', 'valid']

classes = ["nie es8","maybach s650","toyota gt8","tesla modelx"]  #

def load_vim_label(labelfile):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')

    image_list = annotations['_via_image_id_list']
    print ("<<<image ls: ", image_list)
    print (annotations)

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

    print ("<<<< finish rename imgs!")

if __name__ == "__main__":
    # first make sure your images is preprocessed before labeling!
    imgfolder = '/Users/liujunyi/Desktop/spottag/summit-training/道路/pics/imgs'
    preprocess(imgfolder,'../data/custom/images')
    #load_vim_label('../data/custom/labels/car-type.json')

'''
# sets generation
image_list = os.listdir('../data/custom/images')
label_list = os.listdir('../data/custom/labels')

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

'''
