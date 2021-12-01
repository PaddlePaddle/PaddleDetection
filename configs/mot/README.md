简体中文 | [English](README_en.md)

# 多目标跟踪 (Multi-Object Tracking)

## 内容
- [简介](#简介)
- [安装依赖](#安装依赖)
- [模型库](#模型库)
- [数据集准备](#数据集准备)
- [引用](#引用)

## 简介

当前主流的多目标追踪(MOT)算法主要由两部分组成：Detection+Embedding。Detection部分即针对视频，检测出每一帧中的潜在目标。Embedding部分则将检出的目标分配和更新到已有的对应轨迹上(即ReID重识别任务)。根据这两部分实现的不同，又可以划分为**SDE**系列和**JDE**系列算法。

- SDE(Separate Detection and Embedding)这类算法完全分离Detection和Embedding两个环节，最具代表性的就是**DeepSORT**算法。这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是串联的导致速度慢耗时较长，在构建实时MOT系统中面临较大挑战。

- JDE(Joint Detection and Embedding)这类算法完是在一个共享神经网络中同时学习Detection和Embedding，使用一个多任务学习的思路设置损失函数。代表性的算法有**JDE**和**FairMOT**。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。

PaddleDetection实现了这两个系列的3种多目标跟踪算法。
- [DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个ReID重识别任务。DeepSORT所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID模型此处选择[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`PCB+Pyramid ResNet101`和`PPLCNet`模型。

- [JDE](https://arxiv.org/abs/1909.12605)(Joint Detection and Embedding)是在一个单一的共享神经网络中同时学习目标检测任务和embedding任务，并同时输出检测结果和对应的外观embedding匹配的算法。JDE原论文是基于Anchor Base的YOLOv3检测器新增加一个ReID分支学习embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。

- [FairMOT](https://arxiv.org/abs/2004.01888)以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

[PP-Tracking](../../deploy/pptracking/README.md)是基于PaddlePaddle深度学习框架的业界首个开源实时跟踪系统。针对实际业务的难点痛点，PP-Tracking内置行人车辆跟踪、跨镜头跟踪、多类别跟踪、小目标跟踪及流量计数等能力与产业应用，同时提供可视化开发界面。模型集成多目标跟踪，目标检测，ReID轻量级算法，进一步提升PP-Tracking在服务器端部署性能。同时支持python，C++部署，适配Linux，Nvidia Jetson多平台环境。
<div width="1000" align="center">
  <img src="../../docs/images/pptracking.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  视频来源：VisDrone2021, BDD100K开源数据集</div>
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
- 预测需确保已安装[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

## 模型库

- 基础模型
    - [DeepSORT](deepsort/README_cn.md)
    - [JDE](jde/README_cn.md)
    - [FairMOT](fairmot/README_cn.md)
- 特色垂类模型
    - [行人跟踪](pedestrian/README_cn.md)
    - [人头跟踪](headtracking21/README_cn.md)
    - [车辆跟踪](vehicle/README_cn.md)
- 多类别跟踪
    - [多类别跟踪](mcfairmot/README_cn.md)
- 跨境头跟踪
    - [跨境头跟踪](mtmct/README_cn.md)

## 数据集准备

### MOT数据集
PaddleDetection使用和[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) 还有[FairMOT](https://github.com/ifzhang/FairMOT)相同的数据集。请参照[数据准备文档](../../docs/tutorials/PrepareMOTDataSet_cn.md)去下载并准备好所有的数据集包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。使用前6者作为联合数据集参与训练，MOT16作为评测数据集。此外还可以使用**MOT15和MOT20**进行finetune。所有的行人都有检测框标签，部分有ID标签。如果您想使用这些数据集，请**遵循他们的License**。

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
- `identity`是从`1`到`num_identifies`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
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

### 快速下载
按照以下命令可以快速下载各个数据集，注意需要按以上目录解压和存放：
```
wget https://dataset.bj.bcebos.com/mot/MOT17.zip
wget https://dataset.bj.bcebos.com/mot/Caltech.zip
wget https://dataset.bj.bcebos.com/mot/CUHKSYSU.zip
wget https://dataset.bj.bcebos.com/mot/PRW.zip
wget https://dataset.bj.bcebos.com/mot/Cityscapes.zip
wget https://dataset.bj.bcebos.com/mot/ETH.zip
wget https://dataset.bj.bcebos.com/mot/MOT16.zip
```


## 引用
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
