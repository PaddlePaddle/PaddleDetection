English | [简体中文](README.md)

# MOT (Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Model Zoo](#Model_Zoo)
- [Dataset Preparation](#Dataset_Preparation)
- [Citations](#Citations)

## Introduction
The current mainstream of 'Tracking By Detecting' multi-object tracking (MOT) algorithm is mainly composed of two parts: detection and embedding. Detection aims to detect the potential targets in each frame of the video. Embedding assigns and updates the detected target to the corresponding track (named ReID task). According to the different implementation of these two parts, it can be divided into **SDE** series and **JDE** series algorithm.

- **SDE** (Separate Detection and Embedding) is a kind of algorithm which completely separates Detection and Embedding. The most representative is **DeepSORT** algorithm. This design can make the system fit any kind of detectors without difference, and can be improved for each part separately. However, due to the series process, the speed is slow. Time-consuming is a great challenge in the construction of real-time MOT system.
- **JDE** (Joint Detection and Embedding) is to learn detection and embedding simultaneously in a shared neural network, and set the loss function with a multi task learning approach. The representative algorithms are **JDE** and **FairMOT**. This design can achieve high-precision real-time MOT performance.

Paddledetection implements three MOT algorithms of these two series, they are [DeepSORT](https://arxiv.org/abs/1812.00442) of SDE algorithm, and [JDE](https://arxiv.org/abs/1909.12605),[FairMOT](https://arxiv.org/abs/2004.01888) of JDE algorithm.

### PP-Tracking real-time MOT system
In addition, PaddleDetection also provides [PP-Tracking](../../deploy/pptracking/README.md) real-time multi-object tracking system.
PP-Tracking is the first open source real-time Multi-Object Tracking system, and it is based on PaddlePaddle deep learning framework. It has rich models, wide application and high efficiency deployment.

PP-Tracking supports two paradigms: single camera tracking (MOT) and multi-camera tracking (MTMCT). Aiming at the difficulties and pain points of actual business, PP-Tracking provides various MOT functions and applications such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. The deployment method supports API and GUI visual interface, and the deployment language supports Python and C++, The deployment platform environment supports Linux, NVIDIA Jetson, etc.

### AI studio public project tutorial
PP-tracking provides an AI studio public project tutorial. Please refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).

### Python predict and deployment
PP-Tracking supports Python predict and deployment. Please refer to this [doc](../../deploy/pptracking/python/README.md).

### C++ predict and deployment
PP-Tracking supports C++ predict and deployment. Please refer to this [doc](../../deploy/pptracking/cpp/README.md).

### GUI predict and deployment
PP-Tracking supports GUI predict and deployment. Please refer to this [doc](https://github.com/yangyudong2020/PP-Tracking_GUi).

<div width="1000" align="center">
  <img src="../../docs/images/pptracking_en.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  video source：VisDrone, BDD100K dataset</div>
</div>


## Installation
Install all the related dependencies for MOT:
```
pip install lap sklearn motmetrics openpyxl
or
pip install -r requirements.txt
```
**Notes:**
- Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


## Model Zoo
- Base models
    - [ByteTrack](bytetrack/README.md)
    - [DeepSORT](deepsort/README.md)
    - [JDE](jde/README.md)
    - [FairMOT](fairmot/README.md)
- Feature models
    - [Pedestrian](pedestrian/README.md)
    - [Head](headtracking21/README.md)
    - [Vehicle](vehicle/README.md)
- Multi-Class Tracking
    - [MCFairMOT](mcfairmot/README.md)
- Multi-Target Multi-Camera Tracking
    - [MTMCT](mtmct/README.md)


## Dataset Preparation
### MOT Dataset
PaddleDetection implement [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) and [FairMOT](https://github.com/ifzhang/FairMOT), and use the same training data named 'MIX' as them, including **Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16**. The former six are used as the mixed dataset for training, and MOT16 are used as the evaluation dataset. If you want to use these datasets, please **follow their licenses**.

**Notes:**
- Multi-Object Tracking(MOT) datasets are always used for single category tracking. DeepSORT, JDE and FairMOT are single category MOT models. 'MIX' dataset and it's sub datasets are also single category pedestrian tracking datasets. It can be considered that there are additional IDs ground truth for detection datasets.
- In order to train the feature models of more scenes, more datasets are also processed into the same format as the MIX dataset. PaddleDetection Team also provides feature datasets and models of [vehicle tracking](vehicle/readme.md), [head tracking](headtracking21/readme.md) and more general [pedestrian tracking](pedestrian/readme.md). User defined datasets can also be prepared by referring to data preparation [doc](../../docs/tutorials/PrepareMOTDataSet.md).
- The multipe category MOT model is [MCFairMOT] (mcfairmot/readme_cn.md), and the multi category dataset is the integrated version of VisDrone dataset. Please refer to the doc of [MCFairMOT](mcfairmot/README.md).
- The Multi-Target Multi-Camera Tracking (MTMCT) model is [AIC21 MTMCT](https://www.aicitychallenge.org)(CityFlow) Multi-Camera Vehicle Tracking dataset. The dataset and model can refer to the doc of [MTMCT](mtmct/README.md)

### Dataset Directory
First, download the image_lists.zip using the following command, and unzip them into `PaddleDetection/dataset/mot`:
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```

Then, download the MIX dataset using the following command, and unzip them into `PaddleDetection/dataset/mot`:
```
wget https://dataset.bj.bcebos.com/mot/MOT17.zip
wget https://dataset.bj.bcebos.com/mot/Caltech.zip
wget https://dataset.bj.bcebos.com/mot/CUHKSYSU.zip
wget https://dataset.bj.bcebos.com/mot/PRW.zip
wget https://dataset.bj.bcebos.com/mot/Cityscapes.zip
wget https://dataset.bj.bcebos.com/mot/ETHZ.zip
wget https://dataset.bj.bcebos.com/mot/MOT16.zip
```

The final directory is:
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
            |——————mot16.train  
            |——————mot17.train  
            |——————prw.train  
            |——————prw.val
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT16
  |——————MOT17
  |——————PRW
```

### Data Format
These several relevant datasets have the following structure:
```
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
- `class` is the class id, support single class and multi-class, start from `0`, and for single class is `0`.
- `identity` is an integer from `1` to `num_identities`(`num_identities` is the total number of instances of objects in the dataset of all videos or image squences), or `-1` if this box has no identity annotation.
- `[x_center] [y_center] [width] [height]` are the center coordinates, width and height, note that they are normalized by the width/height of the image, so they are floating point numbers ranging from 0 to 1.


## Citations
```
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

@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```
