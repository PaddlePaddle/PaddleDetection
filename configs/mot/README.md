English | [简体中文](README_cn.md)

# MOT (Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Model Zoo](#Model_Zoo)
- [Dataset Preparation](#Dataset_Preparation)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction
PaddleDetection implements three multi-object tracking methods.
- [DeepSORT](https://arxiv.org/abs/1812.00442) (Deep Cosine Metric Learning SORT) extends the original [SORT](https://arxiv.org/abs/1703.07402) (Simple Online and Realtime Tracking) algorithm to integrate appearance information based on a deep appearance descriptor. It adds a CNN model to extract features in image of human part bounded by a detector. Here we use `JDE` as detection model to generate boxes, and select `PCBPyramid` as the ReID model. We also support loading the boxes from saved detection result files.

- [JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding) is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network.

- [FairMOT](https://arxiv.org/abs/2004.01888) focuses on accomplishing the detection and re-identification in a single network to improve the inference speed, presents a simple baseline which consists of two homogeneous branches to predict pixel-wise objectness scores and re-ID features. The achieved fairness between the two tasks allows FairMOT to obtain high levels of detection and tracking accuracy.

<div align="center">
  <img src="../../docs/images/mot16_jde.gif" width=500 />
</div>


## Installation

Install all the related dependencies for MOT:
```
pip install lap sklearn motmetrics openpyxl cython_bbox
or
pip install -r requirements.txt
```
**Notes:**
- Install `cython_bbox` for Windows: `pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox`. You can refer to this [tutorial](https://stackoverflow.com/questions/60349980/is-there-a-way-to-install-cython-bbox-for-windows)
- Evaluation on Windows CUDA 11 environment may not be normally. It will be repaired as soon as possible. You can change to CUDA 10.2 or CUDA 10.1 environment for normal evaluation.


## Model Zoo

### JDE on MOT-16 training set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  73.2  |  69.3  | 1351  |  6591  | 21625 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  70.1  |  65.2  | 1328  |  6441  | 25187 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.2  |  64.5  | 1308  |  7011  | 32252 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**Notes:**
 JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

### DeepSORT on MOT-16 training set

| backbone  | input shape  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | Detector | ReID | config |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:-----: | :-----: | :-----: |
| DarkNet53 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  5.07 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**Notes:**
 DeepSORT does not need to train, only used for evaluation. Before DeepSORT evaluation, you should get detection results by a detection model first, here we use JDE, and then prepare them like this:
```
det_results_dir
   |——————MOT16-02.txt
   |——————MOT16-04.txt
   |——————MOT16-05.txt
   |——————MOT16-09.txt
   |——————MOT16-10.txt
   |——————MOT16-11.txt
   |——————MOT16-13.txt
```
Each txt is the detection result of all the pictures extracted from each video, and each line describes a bounding box with the following format:
```
[frame_id][identity][bb_left][bb_top][width][height][conf][x][y][z]
```
**Notes:**
- `frame_id` is the frame number of the image
- `identity` is the object id using default value `-1`
- `bb_left` is the X coordinate of the left bound of the object box
- `bb_top` is the Y coordinate of the upper bound of the object box
- `width, height` is the pixel width and height
- `conf` is the object score with default value `1` (the results had been filtered out according to the detection score threshold)
- `x,y,z` are used in 3D, default to `-1` in 2D.

### FairMOT Results on MOT-16 train set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |   544  |  3822  |  14095  |     -   |    -   |   -    |
| DLA-34         | 1088x608 |  83.7  |  83.3  |   435  |  3829  |  13764  |     -   | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |


### FairMOT Results on MOT-16 test set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |    -   |    -   |   25.9   |    -   |   -    |
| DLA-34         | 1088x608 |  74.8  |  74.4  |  930   |  7038  |  37994 |    -     | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

**Notes:**
 FairMOT used 8 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.

## Dataset Preparation

### MOT Dataset
PaddleDetection use the same training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) and [FairMOT](https://github.com/ifzhang/FairMOT). Please refer to [PrepareMOTDataSet](../../docs/tutorials/PrepareMOTDataSet.md) to download and prepare all the training data including **Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16**. **MOT15 and MOT20** can also be downloaded from the official webpage of MOT challenge. If you want to use these datasets, please **follow their licenses**.

### Data Format
These several relevant datasets have the following structure:
```
Caltech
   |——————images
   |        └——————00001.jpg
   |        |—————— ...
   |        └——————0000N.jpg
   └——————labels_with_ids
            └——————00001.txt
            |—————— ...
            └——————0000N.txt
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train
```
Annotations of these datasets are provided in a unified format. Every image has a corresponding annotation text. Given an image path, the annotation text path can be generated by replacing the string `images` with `labels_with_ids` and replacing `.jpg` with `.txt`.

In the annotation text, each line is describing a bounding box and has the following format:
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**Notes:**
- `class` should be `0`. Only single-class multi-object tracking is supported now.
- `identity` is an integer from `0` to `num_identities - 1`(`num_identities` is the total number of instances of objects in the dataset), or `-1` if this box has no identity annotation.
- `[x_center] [y_center] [width] [height]` are the center coordinates, width and height, note that they are normalized by the width/height of the image, so they are floating point numbers ranging from 0 to 1.

### Dataset Directory

First, follow the command below to download the `image_list.zip` and unzip it in the `dataset/mot` directory:
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```
Then download and unzip each dataset, and the final directory is as follows:
```
dataset/mot
  |——————image_lists
            |——————caltech.10k.val  
            |——————caltech.all  
            |——————caltech.train  
            |——————caltech.val  
            |——————citypersons.train  
            |——————citypersons.val  
            |——————cuhksysu.train  
            |——————cuhksysu.val  
            |——————eth.train  
            |——————mot15.train  
            |——————mot16.train  
            |——————mot17.train  
            |——————mot20.train  
            |——————prw.train  
            |——————prw.val
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT15
  |——————MOT16
  |——————MOT17
  |——————MOT20
  |——————PRW
```

## Getting Start

### 1. Training

Training FairMOT on 8 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```

### 2. Evaluation

Evaluating the track performance of FairMOT on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```

### 3. Inference

Inference a vidoe on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}

@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
