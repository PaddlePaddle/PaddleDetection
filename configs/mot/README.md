简体中文 | [English](README_en.md)

# 多目标跟踪 (Multi-Object Tracking)

## 内容
- [简介](#简介)
- [安装依赖](#安装依赖)
- [模型库和选型](#模型库和选型)
- [MOT数据集准备](#MOT数据集准备)
    - [SDE数据集](#SDE数据集)
    - [JDE数据集](#JDE数据集)
    - [用户自定义数据集准备](#用户自定义数据集准备)
- [引用](#引用)


## 简介
多目标跟踪(Multi-Object Tracking, MOT)是对给定视频或图片序列，定位出多个感兴趣的目标，并在连续帧之间维持个体的ID信息和记录其轨迹。
当前主流的做法是Tracking By Detecting方式，算法主要由两部分组成：Detection + Embedding。Detection部分即针对视频，检测出每一帧中的潜在目标。Embedding部分则将检出的目标分配和更新到已有的对应轨迹上(即ReID重识别任务)，进行物体间的长时序关联。根据这两部分实现的不同，又可以划分为**SDE**系列和**JDE**系列算法。
- SDE(Separate Detection and Embedding)这类算法完全分离Detection和Embedding两个环节，最具代表性的是**DeepSORT**算法。这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是串联的导致速度慢耗时较长。也有算法如**ByteTrack**算法为了降低耗时，不使用Embedding特征来计算外观相似度，前提是检测器的精度足够高。
- JDE(Joint Detection and Embedding)这类算法完是在一个共享神经网络中同时学习Detection和Embedding，使用一个多任务学习的思路设置损失函数。代表性的算法有**JDE**和**FairMOT**。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。

PaddleDetection中提供了SDE和JDE两个系列的多种算法实现：
- SDE
  - [ByteTrack](./bytetrack)
  - [OC-SORT](./ocsort)
  - [BoT-SORT](./botsort)
  - [DeepSORT](./deepsort)
  - [CenterTrack](./centertrack)
- JDE
  - [JDE](./jde)
  - [FairMOT](./fairmot)
  - [MCFairMOT](./mcfairmot)

**注意：**
  - 以上算法原论文均为单类别的多目标跟踪，PaddleDetection团队同时也支持了[ByteTrack](./bytetrack)和FairMOT([MCFairMOT](./mcfairmot))的多类别的多目标跟踪；
  - [DeepSORT](./deepsort)、[JDE](./jde)、[OC-SORT](./ocsort)、[BoT-SORT](./botsort)和[CenterTrack](./centertrack)均只支持单类别的多目标跟踪；
  - [DeepSORT](./deepsort)需要额外添加ReID权重一起执行，[ByteTrack](./bytetrack)可加可不加ReID权重，默认不加；


### 实时多目标跟踪系统 PP-Tracking
PaddleDetection团队提供了实时多目标跟踪系统[PP-Tracking](../../deploy/pptracking)，是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。
PP-Tracking单镜头跟踪采用的方案是[FairMOT](./fairmot)，跨镜头跟踪采用的方案是[DeepSORT](./deepsort)。

<div width="1000" align="center">
  <img src="../../docs/images/pptracking.png"/>
</div>

<div width="1000" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205546999-f847183d-73e5-4abe-9896-ce6a245efc79.gif"/>
  <br>
  视频来源：VisDrone和BDD100K公开数据集</div>
</div>

#### AI Studio公开项目案例
教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

#### Python端预测部署
教程请参考[PP-Tracking Python部署文档](../../deploy/pptracking/python/README.md)。

#### C++端预测部署
教程请参考[PP-Tracking C++部署文档](../../deploy/pptracking/cpp/README.md)。

#### GUI可视化界面预测部署
教程请参考[PP-Tracking可视化界面使用文档](https://github.com/yangyudong2020/PP-Tracking_GUi)。


### 实时行人分析工具 PP-Human
PaddleDetection团队提供了实时行人分析工具[PP-Human](../../deploy/pipeline)，是基于PaddlePaddle深度学习框架的业界首个开源的产业级实时行人分析工具，具有模型丰富、应用广泛和部署高效三大优势。
PP-Human支持图片/单镜头视频/多镜头视频多种输入方式，功能覆盖多目标跟踪、属性识别、行为分析及人流量计数与轨迹记录。能够广泛应用于智慧交通、智慧社区、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。
PP-Human跟踪采用的方案是[ByteTrack](./bytetrack)。

![](https://user-images.githubusercontent.com/48054808/173030254-ecf282bd-2cfe-43d5-b598-8fed29e22020.gif)

#### AI Studio公开项目案例
PP-Human实时行人分析全流程实战教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3842982)。

PP-Human赋能社区智能精细化管理教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3679564)。



## 安装依赖
一键安装MOT相关的依赖：
```
pip install -r requirements.txt
# 或手动pip安装MOT相关的库
pip install lap motmetrics sklearn
```
**注意：**
  - 预测需确保已安装[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。



## 模型库和选型
- 基础模型
    - [ByteTrack](bytetrack/README_cn.md)
    - [OC-SORT](ocsort/README_cn.md)
    - [BoT-SORT](botsort/README_cn.md)
    - [DeepSORT](deepsort/README_cn.md)
    - [JDE](jde/README_cn.md)
    - [FairMOT](fairmot/README_cn.md)
    - [CenterTrack](centertrack/README_cn.md)
- 特色垂类模型
    - [行人跟踪](pedestrian/README_cn.md)
    - [人头跟踪](headtracking21/README_cn.md)
    - [车辆跟踪](vehicle/README_cn.md)
- 多类别跟踪
    - [多类别跟踪](mcfairmot/README_cn.md)
- 跨境头跟踪
    - [跨境头跟踪](mtmct/README_cn.md)

### 模型选型总结

关于模型选型，PaddleDetection团队提供的总结建议如下：

|    MOT方式      |   经典算法      |  算法流程 |  数据集要求  |  其他特点  |
| :--------------| :--------------| :------- | :----: | :----: |
| SDE系列  | DeepSORT,ByteTrack,OC-SORT,BoT-SORT,CenterTrack | 分离式，两个独立模型权重先检测后ReID，也可不加ReID | 检测和ReID数据相对独立，不加ReID时即纯检测数据集 |检测和ReID可分别调优，鲁棒性较高，AI竞赛常用|
| JDE系列  | FairMOT,JDE | 联合式，一个模型权重端到端同时检测和ReID | 必须同时具有检测和ReID标注 | 检测和ReID联合训练，不易调优，泛化性不强|

**注意：**
  - 由于数据标注的成本较大，建议选型前优先考虑**数据集要求**，如果数据集只有检测框标注而没有ReID标注，是无法使用JDE系列算法训练的，更推荐使用SDE系列；
  - SDE系列算法在检测器精度足够高时，也可以不使用ReID权重进行物体间的长时序关联，可以参照[ByteTrack](bytetrack)；
  - 耗时速度和模型权重参数量计算量有一定关系，耗时从理论上看`不使用ReID的SDE系列 < JDE系列 < 使用ReID的SDE系列`；



## MOT数据集准备
PaddleDetection团队提供了众多公开数据集或整理后数据集的下载链接，参考[数据集下载汇总](DataDownload.md)，用户可以自行下载使用。

根据模型选型总结，MOT数据集可以分为两类：一类纯检测框标注的数据集，仅SDE系列可以使用；另一类是同时有检测和ReID标注的数据集，SDE系列和JDE系列都可以使用。

### SDE数据集
SDE数据集是纯检测标注的数据集，用户自定义数据集可以参照[DET数据准备文档](../../docs/tutorials/data/PrepareDetDataSet.md)准备。

以MOT17数据集为例，下载并解压放在`PaddleDetection/dataset/mot`目录下：
```
wget https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip

```
并修改数据集部分的配置文件如下：
```
num_classes: 1

TrainDataset:
  !COCODataSet
    dataset_dir: dataset/mot/MOT17
    anno_path: annotations/train_half.json
    image_dir: images/train
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    dataset_dir: dataset/mot/MOT17
    anno_path: annotations/val_half.json
    image_dir: images/train

TestDataset:
  !ImageFolder
    dataset_dir: dataset/mot/MOT17
    anno_path: annotations/val_half.json
```

数据集目录为：
```
dataset/mot
        |——————MOT17
                |——————annotations
                |——————images
```

### JDE数据集
JDE数据集是同时有检测和ReID标注的数据集，首先按照以下命令`image_lists.zip`并解压放在`PaddleDetection/dataset/mot`目录下：
```
wget https://bj.bcebos.com/v1/paddledet/data/mot/image_lists.zip
```

然后按照以下命令可以快速下载各个公开数据集，也解压放在`PaddleDetection/dataset/mot`目录下：
```
# MIX数据，同JDE,FairMOT论文使用的数据集
wget https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/Caltech.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/CUHKSYSU.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/PRW.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/Cityscapes.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/ETHZ.zip
wget https://bj.bcebos.com/v1/paddledet/data/mot/MOT16.zip
```
数据集目录为：
```
dataset/mot
  |——————image_lists
            |——————caltech.all  
            |——————citypersons.train  
            |——————cuhksysu.train  
            |——————eth.train  
            |——————mot16.train  
            |——————mot17.train  
            |——————prw.train  
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT16
  |——————MOT17
  |——————PRW
```

#### JDE数据集的格式
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
  - `class`为类别id，支持单类别和多类别，从`0`开始计，单类别即为`0`。
  - `identity`是从`1`到`num_identities`的整数(`num_identities`是数据集中所有视频或图片序列的不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
  - `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意他们的值是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。


**注意：**
  - MIX数据集是[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT)和[FairMOT](https://github.com/ifzhang/FairMOT)原论文使用的数据集，包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。使用前6者作为联合数据集参与训练，MOT16作为评测数据集。如果您想使用这些数据集，请**遵循他们的License**。
  - MIX数据集以及其子数据集都是单类别的行人跟踪数据集，可认为相比于行人检测数据集多了id号的标注。
  - 更多场景的垂类模型例如车辆行人人头跟踪等，垂类数据集也需要处理成与MIX数据集相同的格式，参照[数据集下载汇总](DataDownload.md)、[车辆跟踪](vehicle/README_cn.md)、[人头跟踪](headtracking21/README_cn.md)以及更通用的[行人跟踪](pedestrian/README_cn.md)。
  - 用户自定义数据集可参照[MOT数据集准备教程](../../docs/tutorials/PrepareMOTDataSet_cn.md)去准备。


### 用户自定义数据集准备
用户自定义数据集准备请参考[MOT数据集准备教程](../../docs/tutorials/PrepareMOTDataSet_cn.md)去准备。

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

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}

@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}

@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}

@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
}
```
