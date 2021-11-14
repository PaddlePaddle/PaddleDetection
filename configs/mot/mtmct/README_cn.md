English | [简体中文](README_cn.md)

# MTMCT (Multi-Target Multi-Camera Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
MTMCT (Multi-Target Multi-Camera Tracking) 跨镜头多目标跟踪是对同一场景下的不同摄像头拍摄的视频进行多目标跟踪，是监控视频领域一个非常重要的研究课题。MTMCT预测的是同一场景下的不同摄像头拍摄的视频，因此其方法的效果受相机数量角度和拓扑结构等信息的影响较大，此处提供的是除去相机相关方法后的基础版本的MTMCT算法实现，如果要继续提高效果还需加上专门针对该场景和相机信息设计的后处理算法。此处提供了DeepSORT方案做MTMCT，为了达到实时性选用了轻量级检测器PicoDet和轻量级ReID模型PPLCNet。

## 模型库
### DeepSORT在 AIC21 MTMCT(CityFlow) 车辆跨境跟踪数据集Test集上的结果

|  检测器       |  输入尺度     |  ReID    |  场景   |  MOTA  |  IDF1  | IDP  |  IDR  |  Precision  |  Recall  |  FPS  | 配置文件 |
|  :-----      | :-----      | :----     | :----- | :----  |:-----  |:---  |:----  |:---------   |:-------  |:----  |:------  |
| PicoDet      | 640x640     | PPLCNet   | S06    |    -   |  -     |  -   | -     |  -          |  -       | -     |[配置文件](./deepsort_picodet_pplcnet_aic21mtmct_vehicle.yml) |
| PPYOLOv2     | 640x640     | PPLCNet   | S06    |    -   |  -     |  -   | -     |  -          |  -       | -     |[配置文件](./deepsort_ppyolov2_pplcnet_aic21mtmct_vehicle.yml) |

**注意:**
  S06是AIC21 MTMCT数据集Test集的场景名称，S06场景下有’c041,c042,c043,c044,c045,c046‘共6个摄像头的视频。


## 快速开始
### 1. 训练
使用8卡GPU训练一个PicoDet车辆检测器
```bash
python -m paddle.distributed.launch --log_dir=./picodet_l_esnet_300e_896x896_aic21mtmct_vehicle/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/mtmct/detector/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle.yml
```
**注意:**
 ReID模型此处使用轻量级的PPLCNet，具体可以参照`configs/mot/deepsort/reid/deepsort_pplcnet_vehicle.yml`，是在VERI-Wild车辆重识别数据集训练得到的权重。

### 2. 评估
DeepSORT同时加载检测器和ReID进行评估:
```bash
config=configs/mot/mtmct/deepsort_picodet_pplcnet_aic21mtmct_vehicle.yml
det_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle.pdparams
reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.pdparams

CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval_mot.py -c ${config} --scaled=True -o det_weights=${det_weights} reid_weights=${reid_weights}
```
**注意:**
  默认是评估S01场景下的’c001,c002,c003,c004,c005‘共5个摄像头的视频. 如需换评估数据集可修改`data_root`和`dataset_dir`：
  ```
  EvalMOTDataset:
    !MOTImageFolder
      dataset_dir: dataset/mtmct
      data_root: S01/images
      keep_ori_im: True # set as True in DeepSORT
  ```
  `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。

### 3. 预测
预测一个场景下的不同摄像头拍摄的视频集
```bash
config=configs/mot/mtmct/deepsort_picodet_pplcnet_aic21mtmct_vehicle.yml
det_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle.pdparams
reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.pdparams

CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c ${config} --scaled=True --mtmct_dir={your mtmct scene video folder}  --save_videos -o det_weights=${det_weights} reid_weights=${reid_weights}
```
**注意:**
  请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
  `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
  MTMCT预测的是同一场景下的不同摄像头拍摄的视频，`--mtmct_dir`是场景视频的文件夹名字，里面包含不同摄像头拍摄的视频，视频数量至少为两个。

### 4. 导出模型
Step 1：导出检测模型
```bash
# 导出PicoDet车辆检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mtmct/detector/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle.pdparams
```
Step 2：导出ReID模型
```bash
# 导出PPLCNet车辆ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mtmct/reid/deepsort_pplcnet_aicity_vehicle.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
# 用导出PicoDet车辆检测模型和PPLCNet车辆ReID模型
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/picodet_l_esnet_300e_896x896_aic21mtmct_vehicle/ --reid_model_dir=output_inference/deepsort_pplcnet_aicity_vehicle/ --mtmct_dir={your mtmct scene video folder} --device=GPU --scaled=True --save_mot_txts --save_images
```
**注意:**
  跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)，或`--save_images`表示保存跟踪结果可视化图片。
  `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
  MTMCT预测的是同一场景下的不同摄像头拍摄的视频，`--mtmct_dir`是某个场景的文件夹名字，里面包含该场景不同摄像头拍摄的视频，视频数量至少为两个。


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
