# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ cd SageMaker/yolov3/source
    $ source activate pytorch_p36
    $ pip install -r requirements.txt -i https://opentuna.cn/pypi/web/simple

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

## data preparation
to prepare the labeled data, refer to `data_preparation` folder, with label tools `index.html`, open this file, label your imgs and run `python vim_label.py`

## Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

#### Example Custom model

Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### Prepare Data

The data folder structure as below

Run the following commands to convert 
```
#here note that you need to change the source pictures s3 path in prepare_data.sh
$ cd data/preparation                                 
$ bash prepare_data.sh
```

```shell script

|-- source
    |-- source
        |-- data
            |-- custom
                |-- images #image folder
                |-- labels  #The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.
                |-- classes.names #This file should have one row per class name.
                |-- train.txt #paths to images that will be used as train.
                |-- valid.txt #paths to images that will be used as validation data
        |-- weights
        ...
```

#### Train
To train on the custom dataset run:

```
$ nohup python train.py --model_def config/yolov3-custom.cfg \
  --data_config config/custom.data \
  --pretrained_weights weights/darknet53.conv.74 \
  --epochs 2000 > my.log 2>&1 &
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.


#### Example Training log via ESD BR Code

![pic1](./document/epochloss.png)

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below

```
$ tensorboard --logdir='logs' --port=6006
```

* Go to http://[notbooklocate]/proxy/6006/

![pic1](./document/loss.png)

#### Inference
To infer on the custom dataset run:

```
$ python detect.py --model_def config/yolov3-custom.cfg \
--weights_path checkpoints/yolov3_ckpt_480.pth \
--image_folder data/custom/images \
--class_path data/custom/classes.names
```
![pic1](./document/res.png)

#### Inference on Sagemaker endpoint

first put your model files (`config file`, `model file` and `class.names`) under `endpoint` folder, change the file name if necessary from `predictor.py`, originally file names as (`yolov3-custom.cfg`,`yolov3_ckpt_1580.pth`,`classes.names`), then run the below command 

```
# build ecr image
sh endpoint/build_and_push.sh
# create endpoint
python create endpoint.py
```

client part on invoke the endpoint
```
runtime = boto3.client("sagemaker-runtime",region_name="us-east-2")
tic = time.time()

# imread then resize then save
a = cv2.imread("./endpoint/1.jpeg")
a_resize = cv2.resize(a, (100,100))
cv2.imwrite("./endpoint/test.jpeg",a_resize)


body = b""
with open("./endpoint/test.jpeg", "rb") as fp:
    body = fp.read()

response = runtime.invoke_endpoint(
    EndpointName='yolov3',
    Body=body,
    ContentType='image/jpeg',
)
body = response["Body"].read()

toc = time.time()

print(body.decode())
print(f"elapsed: {(toc - tic) * 1000.0} ms")
```

you can see the result, the invoke took around 400ms-1s from local pc, around 70-90ms from ec2

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
