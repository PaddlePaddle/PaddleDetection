简体中文 | [English](README.md)

# 多目标跟踪 (Multi-Object Tracking)

## 内容
- [简介](#简介)
- [安装依赖](#安装依赖)
- [模型库](#模型库)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介

PaddleDetection实现了3种多目标跟踪方法。
- [DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息。

- [JDE](https://arxiv.org/abs/1909.12605)(Joint Detection and Embedding)是一个快速高性能多目标跟踪器，它是在共享神经网络中同时学习目标检测任务和外观嵌入任务的。

- [FairMOT](https://arxiv.org/abs/2004.01888)着重研究在单个网络中实现检测和ReID以提高推理速度，提出了一种由两个同质分支组成的简单基线来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了高水平的检测和跟踪精度。

<div align="center">
  <img src="../../docs/images/mot16_jde.gif" width=500 />
</div>


## 安装依赖

一键安装MOT相关的依赖：
```
pip install lap sklearn motmetrics openpyxl cython_bbox
或者
pip install -r requirements.txt
```
**注意：**
- `cython_bbox`在windows上安装：`pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox`。可参考这个[教程](https://stackoverflow.com/questions/60349980/is-there-a-way-to-install-cython-bbox-for-windows)。
- Windows cuda 11环境下可能无法正常评估，会尽快修复，可换到cuda 10.2或cuda 10.1环境下进行正常评估。

## 模型库

### JDE在MOT-16 train集上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  检测模型  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  73.2  |  69.3  | 1351  |  6591  | 21625 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  70.1  |  65.2  | 1328  |  6441  | 25187 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.2  |  64.5  | 1308  |  7011  | 32252 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**注意:**
 JDE使用8个GPU进行训练，每个GPU上batch size为4，训练30个epoch。

### DeepSORT在MOT-16 train集上结果

|  骨干网络  | 输入尺寸 | MOTA |  IDF1  |  IDS | FP  |   FN  |   FPS  | 检测模型 | ReID模型 | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:-----: | :-----: | :-----: |
| DarkNet53 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  5.07 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**注意:**
  DeepSORT此处不需要训练MOT数据集，只用于评估。在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，此处使用JDE，然后像这样准备好结果文件:
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
其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id][identity][bb_left][bb_top][width][height][conf][x][y][z]
```
**注意**:
- `frame_id`是图片帧的序号
- `identity`是目标id采用默认值为`-1`
- `bb_left`是目标框的左边界的x坐标
- `bb_top`是目标框的上边界的y坐标
- `width，height`是真实的像素宽高
- `conf`是目标得分设置为`1`(已经按检测的得分阈值筛选出的检测结果)
- `x,y,z`是3D中用到的，在2D中默认为`-1`

### FairMOT在MOT-16 train集上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |  544  |  3822  | 14095 |    -     |   -   |   -   |
| DLA-34         | 1088x608 |  83.7  |  83.3  |  435  |  3829  | 13764 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

### FairMOT在MOT-16 test集上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |    -   |    -   |   25.9   |    -   |   -    |
| DLA-34         | 1088x608 |  74.8  |  74.4  |  930   |  7038  |  37994 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

**注意:**
 FairMOT使用8个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


## 数据集准备

### MOT数据集
PaddleDetection使用和[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) 还有[FairMOT](https://github.com/ifzhang/FairMOT)相同的数据集。请参照[数据准备文档](../../docs/tutorials/PrepareMOTDataSet_cn.md)去下载并准备好所有的数据集包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。此外还可以下载**MOT15和MOT20**数据集，如果您想使用这些数据集，请**遵循他们的License**。

### 数据格式
这几个相关数据集都遵循以下结构：
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
所有数据集的标注是以统一数据格式提供的。各个数据集中每张图片都有相应的标注文本。给定一个图像路径，可以通过将字符串`images`替换为`labels_with_ids`并将`.jpg`替换为`.txt`来生成标注文本路径。在标注文本中，每行都描述一个边界框，格式如下：
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**注意**:
- `class`为`0`，目前仅支持单类别多目标跟踪。
- `identity`是从`0`到`num_identifies-1`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
- `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意他们的值是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。

### 数据集目录

首先按照以下命令下载image_lists.zip并解压放在`dataset/mot`目录下：
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```
然后依次下载各个数据集并解压，最终目录为：
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

## 快速开始

### 1. 训练

FairMOT使用8GPU通过如下命令一键式启动训练

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```

### 2. 评估

FairMOT使用单张GPU通过如下命令一键式启动评估

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```

**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

## 引用
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
