[English](README.md) | 简体中文
# 特色垂类跟踪模型

## 大规模行人跟踪 (Pedestrian Tracking)

行人跟踪的主要应用之一是交通监控。

[PathTrack](https://www.trace.ethz.ch/publications/2017/pathtrack/index.html)包含720个视频序列，有着超过15000个行人的轨迹。包含了街景、舞蹈、体育运动、采访等各种场景的，大部分是移动摄像头拍摄场景。该数据集只有Pedestrian一类标注作为跟踪任务。

[VisDrone](http://aiskyeye.com)是无人机视角拍摄的数据集，是以俯视视角为主。该数据集涵盖不同位置（取自中国数千个相距数千公里的14个不同城市）、不同环境（城市和乡村）、不同物体（行人、车辆、自行车等）和不同密度（稀疏和拥挤的场景）。[VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset)包含56个视频序列用于训练，7个视频序列用于验证。此处针对VisDrone2019-MOT多目标跟踪数据集进行提取，抽取出类别为pedestrian和people的数据组合成一个大的Pedestrian类别。


## 模型库

### FairMOT在各个数据集val-set上Pedestrian类别的结果

|    数据集      |  骨干网络   |  输入尺寸 |  MOTA  |  IDF1  |  FPS   |  下载链接 | 配置文件 |
| :-------------| :-------- | :------- | :----: | :----: | :----: | :-----: |:------: |
|  PathTrack    |   DLA-34  | 1088x608 |  44.9 |    59.3   |    -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_pathtrack.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_pathtrack.yml) |
|  VisDrone     |   DLA-34  | 1088x608 |  49.2 |   63.1 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_pedestrian.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml) |
|  VisDrone     | HRNetv2-W18| 1088x608 |  40.5 |   54.7 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.yml) |
|  VisDrone     | HRNetv2-W18| 864x480 |  38.6 |   50.9 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone_pedestrian.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone_pedestrian.yml) |
|  VisDrone     | HRNetv2-W18| 576x320 |  30.6 |   47.2 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_pedestrian.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_pedestrian.yml) |

**注意:**
 - FairMOT均使用DLA-34为骨干网络，4个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


## 数据集准备和处理

### 1、数据集处理代码说明
代码统一都在tools目录下
```
# visdrone
tools/visdrone/visdrone2mot.py: 生成visdrone_pedestrian据集
```

### 2、visdrone_pedestrian数据集处理
```
# 复制tool/visdrone/visdrone2mot.py到数据集目录下
# 生成visdrone_pedestrian MOT格式的数据，抽取类别classes=1,2 (pedestrian, people)
<<--生成前目录-->>
├── VisDrone2019-MOT-val
│   ├── annotations
│   ├── sequences
│   ├── visdrone2mot.py
<<--生成后目录-->>
├── VisDrone2019-MOT-val
│   ├── annotations
│   ├── sequences
│   ├── visdrone2mot.py
│   ├── visdrone_pedestrian
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── labels_with_ids
│   │   │   ├── train
│   │   │   ├── val
# 执行
python visdrone2mot.py --transMot=True --data_name=visdrone_pedestrian --phase=val
python visdrone2mot.py --transMot=True --data_name=visdrone_pedestrian --phase=train
```

## 快速开始

### 1. 训练
使用2个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608_visdrone_pedestrian/ --gpus 0,1 tools/train.py -c configs/mot/pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml
```

### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_pedestrian.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml -o weights=output/fairmot_dla34_30e_1088x608_visdrone_pedestrian/model_final.pdparams
```

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_pedestrian.pdparams --video_file={your video name}.mp4  --save_videos
```
**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_pedestrian.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_visdrone_pedestrian --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。

## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}

@INPROCEEDINGS{8237302,
author={S. {Manen} and M. {Gygli} and D. {Dai} and L. V. {Gool}},
booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
title={PathTrack: Fast Trajectory Annotation with Path Supervision},
year={2017},
volume={},
number={},
pages={290-299},
doi={10.1109/ICCV.2017.40},
ISSN={2380-7504},
month={Oct},}

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}
```
