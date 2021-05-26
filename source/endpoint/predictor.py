# -*- coding: utf-8 -*-

import os
import json
import boto3
from models import *

from torch.autograd import Variable

from utils.utils import *
from utils.datasets import *

import warnings
import numpy as np
import torch
import cv2
import time
warnings.filterwarnings("ignore",category=FutureWarning)
import matplotlib.pyplot as plt


import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

# Set up model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("yolov3-custom.cfg", img_size=416).to(device)
# Load checkpoint weights
weights_path = 'yolov3_ckpt_1580.pth'
model.load_state_dict(torch.load(weights_path,map_location=device))

model.eval()  # Set in evaluation mode

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    print ("orig shape: ", orig_im.shape)
    orig_im = orig_im[:, :, :3]
    print("orig shape: ", orig_im.shape)


    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    
    return img_, orig_im, dim

def infer(weights_path, cv_img, class_path,device,model):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #print ("<<<< device", device)
    os.makedirs("output", exist_ok=True)

    # Set up model
    #model = Darknet("yolov3-custom.cfg", img_size=416).to(device)
    # Load checkpoint weights
    #model.load_state_dict(torch.load(weights_path,map_location=device))

    #model.eval()  # Set in evaluation mode

    #classes = load_classes(class_path)  # Extracts class labels from file

    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    print("\nPerforming object detection:")

    # Configure input

    inp_dim = int(416)
   
    print (cv_img.shape)
    img, orig_im, dim = prep_image(cv_img, inp_dim)
    img = img.to(device)
    
    prev_time = time.time()
    
    # Get detections
    with torch.no_grad():
        detections = model(Variable(img))
        detections = non_max_suppression(detections, 0.8, 0.4)

    print ('detections: ', detections)
    
    detections = [i.tolist() for i in detections]
    
    
    end = time.time()
    print("\nPerforming object detection time:", end-prev_time)
    
    '''
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Draw bounding boxes and labels of detections
    for i in detections:
        if i is not None:
            # Rescale boxes to original image
            detection = rescale_boxes(i, 416, cv_img.shape[:2])
            unique_labels = detection[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                print("x1, y1, x2, y2: ", (x1, y1), (x2, y2))

                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_img, classes[int(cls_pred)], (x1, y1), font, 1, [225, 255, 0], 2)
    '''
    
    return detections


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    tic = time.time()
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< title: ", flask.request)
    #print ("<<<< flask.request.data.content_type", flask.request.data.content_type)
    print ("<<<< flask.request.content_type", flask.request.content_type)
    
    data = flask.request.data
    print("len(data)={}".format(len(data)))
    data_np = np.fromstring(data, dtype=np.uint8)
    print("data_np.shape={}".format(str(data_np.shape)))
    print(' '.join(['{:x}'.format(d) for d in data_np[:20].tolist()]), flush=True)
    data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
    print ('data_np: ', data_np)
    data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)

    # make inference
    weights_path = 'yolov3_ckpt_1580.pth'
    class_path = 'classes.names'
    
    ## 
    #device = torch.device("cpu")
    #print ("<<<< cpu 推理 device", device)
    #label = infer(weights_path, data_np, class_path,device)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("<<<< gpu 推理 device", device)
    label = infer(weights_path, data_np, class_path,device,model)

    toc = time.time()
    print(f"0 - invocations: {(toc - tic) * 1000.0} ms")

    inference_result = {
            'result': label
        }
    _payload = json.dumps(inference_result, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')

    #if flask.request.content_type == 'image/jpeg':
       

    #else:
     #   return flask.Response(response='This predictor only supports JSON data and JPEG image data',
                 #             status=415, mimetype='text/plain')