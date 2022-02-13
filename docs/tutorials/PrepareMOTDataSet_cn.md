简体中文 | [English](GETTING_STARTED.md)

# 目录
## 多目标跟踪数据集准备
- [MOT数据集](#MOT数据集)
- [数据集目录](#数据集目录)
- [数据格式](#数据格式)
- [用户数据准备](#用户数据准备)
- [引用](#引用)

### MOT数据集
PaddleDetection复现[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) 和[FairMOT](https://github.com/ifzhang/FairMOT)，是使用的和他们相同的MIX数据集，包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。使用前6者作为联合数据集参与训练，MOT16作为评测数据集。如果您想使用这些数据集，请**遵循他们的License**。

**注意：**
- 多目标跟踪数据集一般是用于单类别的多目标跟踪，DeepSORT、JDE和FairMOT均为单类别跟踪模型，MIX数据集以及其子数据集也都是单类别的行人跟踪数据集，可认为相比于行人检测数据集多了id号的标注。
- 为了训练更多场景的垂类模型例如车辆等，垂类数据集也需要处理成与MIX数据集相同的格式，PaddleDetection也提供了[车辆跟踪](../../configs/mot/vehicle/README_cn.md)、[人头跟踪](../../configs/mot/headtracking21/README_cn.md)以及更通用的[行人跟踪](../../configs/mot/pedestrian/README_cn.md)的垂类数据集和模型。用户自定义数据集也可参照本文档准备。
- 多类别跟踪模型是[MCFairMOT](../../configs/mot/mcfairmot/README_cn.md)，多类别数据集是VisDrone数据集的整合版，可参照[MCFairMOT](../../configs/mot/mcfairmot/README_cn.md)的文档说明。
- 跨镜头跟踪模型，是选用的[AIC21 MTMCT](https://www.aicitychallenge.org) (CityFlow)车辆跨镜头跟踪数据集，数据集和模型可参照[跨境头跟踪](../../configs/mot/mtmct/README_cn.md)的文档说明。

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



### 用户数据准备

为了规范地进行训练和评测，用户数据需要转成和MOT-16数据集相同的目录和格式：
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

#### images文件夹
- `gt.txt`是原始标注文件，而训练所用标注是`labels_with_ids`文件夹。
- `img1`文件夹里是按照一定帧率抽好的图片。
- `seqinfo.ini`文件是视频信息描述文件，需要如下格式的信息：
```
[Sequence]
name=MOT16-02
imDir=img1
frameRate=30
seqLength=600
imWidth=1920
imHeight=1080
imExt=.jpg
```

`gt.txt`里是当前视频中所有图片的原始标注文件，每行都描述一个边界框，格式如下：
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


#### labels_with_ids文件夹
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
