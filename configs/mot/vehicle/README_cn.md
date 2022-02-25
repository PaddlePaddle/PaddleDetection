[English](README.md) | 简体中文
# 特色垂类跟踪模型

## 车辆跟踪 (Vehicle Tracking)

车辆跟踪的主要应用之一是交通监控。在监控场景中，大多是从公共区域的监控摄像头视角拍摄车辆，获取图像后再进行车辆检测和跟踪。


[BDD100K](https://www.bdd100k.com)是伯克利大学AI实验室（BAIR）提出的一个驾驶视频数据集，是以驾驶员视角为主。该数据集不仅分多类别标注，还分晴天、多云等六种天气，住宅区、公路等六种场景，白天、夜晚等三个时间段，以及是否遮挡、是否截断。BDD100K MOT数据集包含1400个视频序列用于训练，200个视频序列用于验证。每个视频序列大约40秒长，每秒5帧，因此每个视频大约有200帧。此处针对BDD100K MOT数据集进行提取，抽取出类别为car, truck, bus, trailer, other vehicle的数据组合成一个Vehicle类别。

[KITTI](http://www.cvlibs.net/datasets/kitti)是一个包含市区、乡村和高速公路等场景采集的数据集，每张图像中最多达15辆车和30个行人，还有各种程度的遮挡与截断。[KITTI-Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)(2D bounding-boxes)数据集一共有50个视频序列，21个为训练集，29个为测试集，目标是估计类别Car和Pedestrian的目标轨迹，此处抽取出类别为Car的数据作为一个Vehicle类别。

[VisDrone](http://aiskyeye.com)是无人机视角拍摄的数据集，是以俯视视角为主。该数据集涵盖不同位置（取自中国数千个相距数千公里的14个不同城市）、不同环境（城市和乡村）、不同物体（行人、车辆、自行车等）和不同密度（稀疏和拥挤的场景）。[VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset)包含56个视频序列用于训练，7个视频序列用于验证。此处针对VisDrone2019-MOT多目标跟踪数据集进行提取，抽取出类别为car、van、truck、bus的数据组合成一个Vehicle类别。


## 模型库

### FairMOT在各个数据集val-set上Vehicle类别的结果

|    数据集      |  骨干网络   |  输入尺寸 |  MOTA  |  IDF1  |  FPS   |  下载链接 | 配置文件 |
| :-------------| :-------- | :------- | :----: | :----: | :----: | :-----: |:------: |
|  BDD100K      |   DLA-34  | 1088x608 |  43.5 |  50.0  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml) |
|  BDD100K      | HRNetv2-W18| 576x320 |  32.6 |  38.7  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.yml) |
|  KITTI        |   DLA-34  | 1088x608 |  82.7 |    -   |    -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_kitti_vehicle.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_kitti_vehicle.yml) |
|  VisDrone     |   DLA-34  | 1088x608 |  52.1 |   63.3 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_vehicle.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_visdrone_vehicle.yml) |
|  VisDrone     | HRNetv2-W18| 1088x608 |  46.0 |   56.8 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle.yml) |
|  VisDrone     | HRNetv2-W18| 864x480 |  43.7 |   56.1 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone_vehicle.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone_vehicle.yml) |
|  VisDrone     | HRNetv2-W18| 576x320 |  39.8 |   52.4 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.yml) |

**注意:**
 - FairMOT均使用DLA-34为骨干网络，4个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


## 数据集准备和处理

### 1、数据集处理代码说明
代码统一都在tools目录下
```
# bdd100kmot
tools/bdd100kmot/gen_bdd100kmot_vehicle.sh：通过执行bdd100k2mot.py和gen_labels_MOT.py生成bdd100kmot_vehicle 数据集
tools/bdd100kmot/bdd100k2mot.py：将bdd100k全集转换成mot格式
tools/bdd100kmot/gen_labels_MOT.py：生成单类别的labels_with_ids文件
# visdrone
tools/visdrone/visdrone2mot.py：生成visdrone_vehicle
```

### 2、bdd100kmot_vehicle数据集处理
```
# 复制tools/bdd100kmot里的代码到数据集目录下
# 生成bdd100kmot_vehicle MOT格式的数据，抽取类别classes=2,3,4,9,10 (car, truck, bus, trailer, other vehicle)
<<--生成前目录-->>
├── bdd100k
│   ├── images
│   ├── labels
<<--生成后目录-->>
├── bdd100k
│   ├── images
│   ├── labels
│   ├── bdd100kmot_vehicle
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── labels_with_ids
│   │   │   ├── train
│   │   │   ├── val
# 执行
sh gen_bdd100kmot_vehicle.sh
```

### 3、visdrone_vehicle数据集处理
```
# 复制tools/visdrone/visdrone2mot.py到数据集目录下
# 生成visdrone_vehicle MOT格式的数据，抽取类别classes=4,5,6,9 (car, van, truck, bus)
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
│   ├── visdrone_vehicle
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── labels_with_ids
│   │   │   ├── train
│   │   │   ├── val
# 执行
python visdrone2mot.py --transMot=True --data_name=visdrone_vehicle --phase=val
python visdrone2mot.py --transMot=True --data_name=visdrone_vehicle --phase=train
```

## 快速开始

### 1. 训练
使用2个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608_bdd100kmot_vehicle/ --gpus 0,1 tools/train.py -c configs/mot/vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml
```

### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml -o weights=output/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle/model_final.pdparams
```

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.pdparams --video_file={your video name}.mp4  --save_videos
```
**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle --video_file={your video name}.mp4 --device=GPU --save_mot_txts
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

@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}

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
