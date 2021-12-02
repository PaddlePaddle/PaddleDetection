English | [简体中文](README_cn.md)

# MTMCT (Multi-Target Multi-Camera Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
MTMCT (Multi-Target Multi-Camera Tracking) 跨镜头多目标跟踪是某一场景下的不同摄像头拍摄的视频进行多目标跟踪，是跟踪领域一个非常重要的研究课题，在安防监控、自动驾驶、智慧城市等行业起着重要作用。MTMCT预测的是同一场景下的不同摄像头拍摄的视频，其方法的效果受场景先验知识和相机数量角度拓扑结构等信息的影响较大，PaddleDetection此处提供的是去除场景和相机相关优化方法后的一个基础版本的MTMCT算法实现，如果要继续提高效果，需要专门针对该场景和相机信息设计后处理算法。此处选用DeepSORT方案做MTMCT，为了达到实时性选用了PaddleDetection自研的[PP-YOLOv2](../../ppyolo/)和轻量级网络[PP-PicoDet](../../picodet/)作为检测器，选用PaddleClas自研的轻量级网络PP-LCNet作为ReID模型。

MTMCT是[PP-Tracking](../../../deploy/pptracking)项目中一个非常重要的方向，[PP-Tracking](../../../deploy/pptracking/README.md)是基于PaddlePaddle深度学习框架的业界首个开源实时跟踪系统。针对实际业务的难点痛点，PP-Tracking内置行人车辆跟踪、跨镜头跟踪、多类别跟踪、小目标跟踪及流量计数等能力与产业应用，同时提供可视化开发界面。模型集成目标检测、轻量级ReID、多目标跟踪等算法，进一步提升PP-Tracking在服务器端部署性能。同时支持Python、C++部署，适配Linux、NVIDIA Jetson等多个平台环境。具体可前往该目录使用。

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

## 模型库
### DeepSORT在 AIC21 MTMCT(CityFlow) 车辆跨境跟踪数据集Test集上的结果

|  检测器       |  输入尺度     |  ReID    |  场景   |  Tricks |  IDF1  |   IDP   |   IDR  | Precision |  Recall  |  FPS  | 检测器下载链接 | ReID下载链接 |
|  :---------  | :---------  | :-------  | :----- | :------ |:-----  |:------- |:-----  |:--------- |:-------- |:----- |:------  | :------  |
| PP-PicoDet   | 640x640     | PP-LCNet  | S06    |    -    | 0.3617 | 0.4417  | 0.3062 |   0.6266  | 0.4343   | -     |[Detector](https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_640_aic21mtmct_vehicle.tar)    |[ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.tar) |
| PPYOLOv2     | 640x640     | PP-LCNet  | S06    |    -    | 0.4450 | 0.4611  | 0.4300 |   0.6385  | 0.5954   | -     |[Detector](https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyolov2_r50vd_dcn_365e_aic21mtmct_vehicle.tar)   |[ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.tar) |

**注意:**
 - S06是AIC21 MTMCT数据集Test集的场景名称，S06场景下有’c041,c042,c043,c044,c045,c046‘共6个摄像头的视频。
 - 由于在部署过程中只需要前向参数，此处提供的是已经导出的模型，解压后可看到包括`infer_cfg.yml`、`model.pdiparams`、`model.pdiparams.info`和`model.pdmodel`四个文件。


## 数据集准备
对于车辆跨镜头跟踪是选用的[AIC21 MTMCT](https://www.aicitychallenge.org) (CityFlow)车辆跨境跟踪数据集，此处提供PaddleDetection团队整理过后的数据集的下载链接：`wget https://paddledet.bj.bcebos.com/data/mot/aic21mtmct_vehicle.zip`，测试使用的是其中的S06文件夹目录，此外还提供AIC21 MTMCT数据集中S01场景抽出来的极小的一个demo测试数据集：`wget https://paddledet.bj.bcebos.com/data/mot/demo/mtmct-demo.tar`

数据集的处理如下所示：
```
# AIC21 MTMCT原始数据集的目录如下所示：
|——————AIC21_Track3_MTMC_Tracking
        |——————cam_framenum  (Number of frames below each camera)  
        |——————cam_loc  （Positional relationship between cameras）
        |——————cam_timestamp  (Time difference between cameras)  
        |——————eval  (evaluation function and ground_truth.txt)
        |——————test  (S06 dataset)
        |——————train  (S01,S03,S04 dataset)
        |——————validation  (S02,S05 dataset)
        |——————DataLicenseAgreement_AICityChallenge_2021.pdf  
        |——————list_cam.txt  (List of all camera paths)
        |——————ReadMe.txt  (Dataset description)
|——————gen_aicity_mtmct_data.py (Camera videos extraction script)
```
需要处理成如下格式：
```
aic21mtmct_vehicle/
├── S01
    ├── gt
    │    ├── gt.txt
    ├── images
         ├── c001
         │    ├── img1
         │    │     ├── 0000001.jpg
         │    │     ...
         │    ├── roi.jpg
         ├── c002
         ...
         ├── c006
├── S02
...
├── S05
├── S06
    ├── images
         ├── c041
              ├── img1
                   ├── 0000001.jpg
                    ...

         ├── c042
         ...
         ├── c046
    ├── zone (only for test-set S06 when use camera tricks for testing)
         ├── c041.png
         ...
         ├── c046.png
```

#### 生成S01场景的验证集数据
python gen_aicity_mtmct_data.py ./AIC21_Track3_MTMC_Tracking/train/S01

**注意:**
 - AIC21 MTMCT数据集共有6个场景共计46个摄像头的数据，其中S01、S03和S04为训练集，S02和S05为验证集，S06是测试集，S06场景下有’c041,c042,c043,c044,c045,c046‘共6个摄像头的视频。

## 快速开始

### 1. 导出模型
Step 1：下载导出的检测模型
```bash
wget https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_640_aic21mtmct_vehicle.tar
tar -xvf picodet_l_640_aic21mtmct_vehicle.tar
```
Step 2：下载导出的ReID模型
```bash
wget https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.tar
tar -xvf deepsort_pplcnet_vehicle.tar
```
**注意:**
 - PP-PicoDet是轻量级检测模型，其训练请参考[configs/picodet](../../picodet/README.md)，并注意修改种类数和数据集路径。
 - PP-LCNet是轻量级ReID模型，其训练请参考[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)，是在VERI-Wild车辆重识别数据集训练得到的权重，建议直接使用无需重训。


### 2. 用导出的模型基于Python去预测
```bash
# 下载demo测试视频
wget https://paddledet.bj.bcebos.com/data/mot/demo/mtmct-demo.tar
tar -xvf mtmct-demo.tar

# 用导出的PicoDet车辆检测模型和PPLCNet车辆ReID模型去基于Python预测
python deploy/pptracking/python/mot_sde_infer.py --model_dir=picodet_l_640_aic21mtmct_vehicle/ --reid_model_dir=deepsort_pplcnet_vehicle/ --mtmct_dir=mtmct-demo --mtmct_cfg=mtmct_cfg --device=GPU --scaled=True --save_mot_txts --save_images
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)，或`--save_images`表示保存跟踪结果可视化图片。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
 - `--mtmct_dir`是MTMCT预测的某个场景的文件夹名字，里面包含该场景不同摄像头拍摄视频的图片文件夹视频，其数量至少为两个。
 - `--mtmct_cfg`是MTMCT预测的某个场景的配置文件，里面包含该一些trick操作的开关和该场景摄像头相关设置的文件路径，用户可以自行更改相关路径以及设置某些操作是否启用。
 - MTMCT跨镜头跟踪输出结果为视频和txt形式。每个图片文件夹各生成一个可视化的跨镜头跟踪结果，与单镜头跟踪的结果是不同的，单镜头跟踪的结果在几个视频文件夹间是独立无关的。MTMCT的结果txt只有一个，比单镜头跟踪结果txt多了第一列镜头id号，跨镜头跟踪结果txt文件每行信息是`camera_id,frame,id,x1,y1,w,h,-1,-1`。
 - MTMCT是[PP-Tracking](../../../deploy/pptracking)项目中的一个非常重要的方向，具体可前往该目录使用。


## 引用
```
@InProceedings{Tang19CityFlow,
author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},
title = {CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019},
pages = {8797–8806}
}
```
