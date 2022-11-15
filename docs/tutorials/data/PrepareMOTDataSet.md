简体中文 | [English](PrepareMOTDataSet_en.md)

# 多目标跟踪数据集准备
## 目录
- [简介和模型选型](#简介和模型选型)
- [MOT数据集准备](#MOT数据集准备)
    - [SDE数据集](#SDE数据集)
    - [JDE数据集](#JDE数据集)
- [用户自定义数据集准备](#用户自定义数据集准备)
    - [SDE数据集](#SDE数据集)
    - [JDE数据集](#JDE数据集)
- [引用](#引用)

## 简介和模型选型
PaddleDetection中提供了SDE和JDE两个系列的多种算法实现：
- SDE(Separate Detection and Embedding)
    - [ByteTrack](../../../configs/mot/bytetrack)
    - [DeepSORT](../../../configs/mot/deepsort)

- JDE(Joint Detection and Embedding)
    - [JDE](../../../configs/mot/jde)
    - [FairMOT](../../../configs/mot/fairmot)
    - [MCFairMOT](../../../configs/mot/mcfairmot)

**注意：**
  - 以上算法原论文均为单类别的多目标跟踪，PaddleDetection团队同时也支持了[ByteTrack](./bytetrack)和FairMOT([MCFairMOT](./mcfairmot))的多类别的多目标跟踪；
  - [DeepSORT](../../../configs/mot/deepsort)和[JDE](../../../configs/mot/jde)均只支持单类别的多目标跟踪；
  - [DeepSORT](../../../configs/mot/deepsort)需要额外添加ReID权重一起执行，[ByteTrack](../../../configs/mot/bytetrack)可加可不加ReID权重，默认不加；


关于模型选型，PaddleDetection团队提供的总结建议如下：

|    MOT方式      |   经典算法      |  算法流程 |  数据集要求  |  其他特点  |
| :--------------| :--------------| :------- | :----: | :----: |
| SDE系列  | DeepSORT,ByteTrack | 分离式，两个独立模型权重先检测后ReID，也可不加ReID | 检测和ReID数据相对独立，不加ReID时即纯检测数据集 |检测和ReID可分别调优，鲁棒性较高，AI竞赛常用|
| JDE系列  | FairMOT | 联合式，一个模型权重端到端同时检测和ReID | 必须同时具有检测和ReID标注 | 检测和ReID联合训练，不易调优，泛化性不强|

**注意：**
  - 由于数据标注的成本较大，建议选型前优先考虑**数据集要求**，如果数据集只有检测框标注而没有ReID标注，是无法使用JDE系列算法训练的，更推荐使用SDE系列；
  - SDE系列算法在检测器精度足够高时，也可以不使用ReID权重进行物体间的长时序关联，可以参照[ByteTrack](bytetrack)；
  - 耗时速度和模型权重参数量计算量有一定关系，耗时从理论上看`不使用ReID的SDE系列 < JDE系列 < 使用ReID的SDE系列`；


## MOT数据集准备
PaddleDetection团队提供了众多公开数据集或整理后数据集的下载链接，参考[数据集下载汇总](../../../configs/mot/DataDownload.md)，用户可以自行下载使用。

根据模型选型总结，MOT数据集可以分为两类：一类纯检测框标注的数据集，仅SDE系列可以使用；另一类是同时有检测和ReID标注的数据集，SDE系列和JDE系列都可以使用。

### SDE数据集
SDE数据集是纯检测标注的数据集，用户自定义数据集可以参照[DET数据准备文档](./PrepareDetDataSet.md)准备。

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


## 用户自定义数据集准备

### SDE数据集
如果用户选择SDE系列方案，是准备准检测标注的自定义数据集，则可以参照[DET数据准备文档](./PrepareDetDataSet.md)准备。

### JDE数据集
如果用户选择JDE系列方案，则需要同时具有检测和ReID标注，且符合MOT-17数据集的格式。
为了规范地进行训练和评测，用户数据需要转成和MOT-17数据集相同的目录和格式：
```
custom_data
   |——————images
   |        └——————test
   |        └——————train
   |                └——————seq1
   |                |        └——————gt
   |                |        |       └——————gt.txt
   |                |        └——————img1
   |                |        |       └——————000001.jpg
   |                |        |       |——————000002.jpg
   |                |        |       └—————— ...
   |                |        └——————seqinfo.ini
   |                └——————seq2
   |                └——————...
   └——————labels_with_ids
            └——————train
                    └——————seq1
                    |        └——————000001.txt
                    |        |——————000002.txt
                    |        └—————— ...
                    └——————seq2
                    └—————— ...
```

##### images文件夹
  - `gt.txt`是原始标注文件，而训练所用标注是`labels_with_ids`文件夹。
  - `gt.txt`里是当前视频中所有图片的原始标注文件，每行都描述一个边界框，格式如下：
    ```
    [frame_id],[identity],[bb_left],[bb_top],[width],[height],[score],[label],[vis_ratio]
    ```
  - `img1`文件夹里是按照一定帧率抽好的图片。
  - `seqinfo.ini`文件是视频信息描述文件，需要如下格式的信息：
    ```
    [Sequence]
    name=MOT17-02
    imDir=img1
    frameRate=30
    seqLength=600
    imWidth=1920
    imHeight=1080
    imExt=.jpg
    ```

其中`gt.txt`里是当前视频中所有图片的原始标注文件，每行都描述一个边界框，格式如下：
```
[frame_id],[identity],[bb_left],[bb_top],[width],[height],[score],[label],[vis_ratio]
```
**注意**:
  - `frame_id`为当前图片帧序号
  - `identity`是从`1`到`num_identities`的整数(`num_identities`是**当前视频或图片序列**的不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
  - `bb_left`是目标框的左边界的x坐标
  - `bb_top`是目标框的上边界的y坐标
  - `width，height`是真实的像素宽高
  - `score`是当前目标是否进入考虑范围内的标志(值为0表示此目标在计算中被忽略，而值为1则用于将其标记为活动实例)，默认为`1`
  - `label`是当前目标的种类标签，由于目前仅支持单类别跟踪，默认为`1`，MOT-16数据集中会有其他类别标签，但都是当作ignore类别计算
  - `vis_ratio`是当前目标被其他目标包含或覆挡后的可见率，是从0到1的浮点数，默认为`1`


##### labels_with_ids文件夹
所有数据集的标注是以统一数据格式提供的。各个数据集中每张图片都有相应的标注文本。给定一个图像路径，可以通过将字符串`images`替换为`labels_with_ids`并将`.jpg`替换为`.txt`来生成标注文本路径。在标注文本中，每行都描述一个边界框，格式如下：
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**注意**:
  - `class`为类别id，支持单类别和多类别，从`0`开始计，单类别即为`0`。
  - `identity`是从`1`到`num_identities`的整数(`num_identities`是数据集中所有视频或图片序列的不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
  - `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。

可采用如下脚本生成相应的`labels_with_ids`:
```
cd dataset/mot
python gen_labels_MOT.py
```


### 引用
Caltech:
```
@inproceedings{ dollarCVPR09peds,
       author = "P. Doll\'ar and C. Wojek and B. Schiele and  P. Perona",
       title = "Pedestrian Detection: A Benchmark",
       booktitle = "CVPR",
       month = "June",
       year = "2009",
       city = "Miami",
}
```
Citypersons:
```
@INPROCEEDINGS{Shanshan2017CVPR,
  Author = {Shanshan Zhang and Rodrigo Benenson and Bernt Schiele},
  Title = {CityPersons: A Diverse Dataset for Pedestrian Detection},
  Booktitle = {CVPR},
  Year = {2017}
 }

@INPROCEEDINGS{Cordts2016Cityscapes,
title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2016}
}
```
CUHK-SYSU:
```
@inproceedings{xiaoli2017joint,
  title={Joint Detection and Identification Feature Learning for Person Search},
  author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
  booktitle={CVPR},
  year={2017}
}
```
PRW:
```
@inproceedings{zheng2017person,
  title={Person re-identification in the wild},
  author={Zheng, Liang and Zhang, Hengheng and Sun, Shaoyan and Chandraker, Manmohan and Yang, Yi and Tian, Qi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1367--1376},
  year={2017}
}
```
ETHZ:
```
@InProceedings{eth_biwi_00534,
author = {A. Ess and B. Leibe and K. Schindler and and L. van Gool},
title = {A Mobile Vision System for Robust Multi-Person Tracking},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR'08)},
year = {2008},
month = {June},
publisher = {IEEE Press},
keywords = {}
}
```
MOT-16&17:
```
@article{milan2016mot16,
  title={MOT16: A benchmark for multi-object tracking},
  author={Milan, Anton and Leal-Taix{\'e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
```
