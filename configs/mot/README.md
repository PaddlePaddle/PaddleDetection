简体中文 | [English](README_en.md)

# 多目标跟踪 (Multi-Object Tracking)

## 内容
- [简介](#简介)
- [安装依赖](#安装依赖)
- [模型库](#模型库)
- [数据集准备](#数据集准备)
- [引用](#引用)


## 简介
当前主流的Tracking By Detecting方式的多目标追踪(Multi-Object Tracking, MOT)算法主要由两部分组成：Detection+Embedding。Detection部分即针对视频，检测出每一帧中的潜在目标。Embedding部分则将检出的目标分配和更新到已有的对应轨迹上(即ReID重识别任务)。根据这两部分实现的不同，又可以划分为**SDE**系列和**JDE**系列算法。
- SDE(Separate Detection and Embedding)这类算法完全分离Detection和Embedding两个环节，最具代表性的就是**DeepSORT**算法。这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是串联的导致速度慢耗时较长，在构建实时MOT系统中面临较大挑战。
- JDE(Joint Detection and Embedding)这类算法完是在一个共享神经网络中同时学习Detection和Embedding，使用一个多任务学习的思路设置损失函数。代表性的算法有**JDE**和**FairMOT**。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。

PaddleDetection实现了这两个系列的3种多目标跟踪算法，分别是SDE系列的[DeepSORT](https://arxiv.org/abs/1812.00442)和JDE系列的[JDE](https://arxiv.org/abs/1909.12605)与[FairMOT](https://arxiv.org/abs/2004.01888)。

### PP-Tracking 实时多目标跟踪系统
此外，PaddleDetection还提供了[PP-Tracking](../../deploy/pptracking/README.md)实时多目标跟踪系统。PP-Tracking是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

### Python端预测部署
PP-Tracking 支持Python预测部署，教程请参考[PP-Tracking Python部署文档](../../deploy/pptracking/python/README.md)。

### C++端预测部署
PP-Tracking 支持C++预测部署，教程请参考[PP-Tracking C++部署文档](../../deploy/pptracking/cpp/README.md)。

### GUI可视化界面预测部署
PP-Tracking 提供了简洁的GUI可视化界面，教程请参考[PP-Tracking可视化界面试用版使用文档](https://github.com/yangyudong2020/PP-Tracking_GUi)。

<div width="1000" align="center">
  <img src="../../docs/images/pptracking.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  视频来源：VisDrone和BDD100K公开数据集</div>
</div>


## 安装依赖
一键安装MOT相关的依赖：
```
pip install lap sklearn motmetrics openpyxl
或者
pip install -r requirements.txt
```
**注意：**
- 预测需确保已安装[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。


## 模型库
- 基础模型
    - [ByteTrack](bytetrack/README_cn.md)
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
PaddleDetection复现[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) 和[FairMOT](https://github.com/ifzhang/FairMOT)，是使用的和他们相同的MIX数据集，包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。使用前6者作为联合数据集参与训练，MOT16作为评测数据集。如果您想使用这些数据集，请**遵循他们的License**。

**注意：**
- 多目标跟踪数据集一般是用于单类别的多目标跟踪，DeepSORT、JDE和FairMOT均为单类别跟踪模型，MIX数据集以及其子数据集也都是单类别的行人跟踪数据集，可认为相比于行人检测数据集多了id号的标注。
- 为了训练更多场景的垂类模型例如车辆等，垂类数据集也需要处理成与MIX数据集相同的格式，PaddleDetection也提供了[车辆跟踪](vehicle/README_cn.md)、[人头跟踪](headtracking21/README_cn.md)以及更通用的[行人跟踪](pedestrian/README_cn.md)的垂类数据集和模型。用户自定义数据集也可参照[数据准备文档](../../docs/tutorials/PrepareMOTDataSet_cn.md)去准备。
- 多类别跟踪模型是[MCFairMOT](mcfairmot/README_cn.md)，多类别数据集是VisDrone数据集的整合版，可参照[MCFairMOT](mcfairmot/README_cn.md)的文档说明。
- 跨镜头跟踪模型，是选用的[AIC21 MTMCT](https://www.aicitychallenge.org) (CityFlow)车辆跨镜头跟踪数据集，数据集和模型可参照[跨境头跟踪](mtmct/README_cn.md)的文档说明。

### 数据集目录
首先按照以下命令下载image_lists.zip并解压放在`PaddleDetection/dataset/mot`目录下：
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```

然后按照以下命令可以快速下载MIX数据集的各个子数据集，并解压放在`PaddleDetection/dataset/mot`目录下：
```
wget https://dataset.bj.bcebos.com/mot/MOT17.zip
wget https://dataset.bj.bcebos.com/mot/Caltech.zip
wget https://dataset.bj.bcebos.com/mot/CUHKSYSU.zip
wget https://dataset.bj.bcebos.com/mot/PRW.zip
wget https://dataset.bj.bcebos.com/mot/Cityscapes.zip
wget https://dataset.bj.bcebos.com/mot/ETHZ.zip
wget https://dataset.bj.bcebos.com/mot/MOT16.zip
```

最终目录为：
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

### 数据格式
这几个相关数据集都遵循以下结构：
```
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
- `class`为类别id，支持单类别和多类别，从`0`开始计，单类别即为`0`。
- `identity`是从`1`到`num_identities`的整数(`num_identities`是数据集中所有视频或图片序列的不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
- `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意他们的值是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。


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
