简体中文 | [English](README.md)

# 多目标跟踪 (Multi-Object Tracking)

## 内容
- [简介](#简介)
- [安装依赖](#安装依赖)
- [模型库](#模型库)
- [特色垂类跟踪模型](#特色垂类跟踪模型)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介

当前主流的多目标追踪(MOT)算法主要由两部分组成：Detection+Embedding。Detection部分即针对视频，检测出每一帧中的潜在目标。Embedding部分则将检出的目标分配和更新到已有的对应轨迹上(即ReID重识别任务)。根据这两部分实现的不同，又可以划分为**SDE**系列和**JDE**系列算法。

- SDE(Separate Detection and Embedding)这类算法完全分离Detection和Embedding两个环节，最具代表性的就是**DeepSORT**算法。这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是串联的导致速度慢耗时较长，在构建实时MOT系统中面临较大挑战。

- JDE(Joint Detection and Embedding)这类算法完是在一个共享神经网络中同时学习Detection和Embedding，使用一个多任务学习的思路设置损失函数。代表性的算法有**JDE**和**FairMOT**。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。

PaddleDetection实现了这两个系列的3种多目标跟踪算法。
- [DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个ReID重识别任务。DeepSORT所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID模型此处选择[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`PCB+Pyramid ResNet101`和`PPLCNet`模型。

- [JDE](https://arxiv.org/abs/1909.12605)(Joint Detection and Embedding)是在一个单一的共享神经网络中同时学习目标检测任务和embedding任务，并同时输出检测结果和对应的外观embedding匹配的算法。JDE原论文是基于Anchor Base的YOLOv3检测器新增加一个ReID分支学习embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。

- [FairMOT](https://arxiv.org/abs/2004.01888)以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

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
- 预测需确保已安装[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

## 模型库

### DeepSORT在MOT-16 Training Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----:| :-----: | :-----: |
| ResNet-101 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort/reid/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  68.3  |  56.5  | 1722 |  17337 | 15890 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort/deepsort_jde_yolov3_pcb_pyramid.yml) |
| PPLCNet    | 1088x608 |  72.2  |  59.5  | 1087  |  8034  | 21481 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort/reid/deepsort_pplcnet.yml) |
| PPLCNet    | 1088x608 |  68.1  |  53.6  | 1979 |  17446 | 15766 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort/deepsort_jde_yolov3_pplcnet.yml) |

### DeepSORT在MOT-16 Test Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----: | :-----: |:-----: |
| ResNet-101 | 1088x608 |  64.1  |  53.0  | 1024  |  12457  | 51919 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) | [ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort/reid/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  61.2  |  48.5  | 1799  |  25796  | 43232 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams)  |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort/deepsort_jde_yolov3_pcb_pyramid.yml) |
| PPLCNet    | 1088x608 |  64.0  |  51.3  | 1208  |  12697  | 51784 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort/reid/deepsort_pplcnet.yml) |
| PPLCNet    | 1088x608 |  61.1  |  48.8  | 2010 |  25401 | 43432 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort/deepsort_jde_yolov3_pplcnet.yml) |

**注意:**
DeepSORT不需要训练MOT数据集，只用于评估，现在支持两种评估的方式。
- **方式1**：加载检测结果文件和ReID模型，在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，然后像这样准备好结果文件:
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
对于MOT16数据集，可以下载PaddleDetection提供的一个经过匹配之后的检测框结果det_results_dir.zip并解压：
```
wget https://dataset.bj.bcebos.com/mot/det_results_dir.zip
```
如果使用更强的检测模型，可以取得更好的结果。其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id],[x0],[y0],[w],[h],[score],[class_id]
```
- `frame_id`是图片帧的序号
- `x0,y0`是目标框的左上角x和y坐标
- `w,h`是目标框的像素宽高
- `score`是目标框的得分
- `class_id`是目标框的类别，如果只有1类则是`0`

- **方式2**：同时加载检测模型和ReID模型，此处选用JDE版本的YOLOv3，具体配置见`configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml`。加载其他通用检测模型可参照`configs/mot/deepsort/deepsort_ppyolov2_pplcnet.yml`进行修改。


### JDE在MOT-16 Training Set上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  下载链接  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  72.0  |  66.9  | 1397  |  7274  | 22209 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](./jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  69.1  |  64.7  | 1539  |  7544  | 25046 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](./jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.7  |  64.4  | 1310  |  6782  | 31964 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](./jde/jde_darknet53_30e_576x320.yml) |


### JDE在MOT-16 Test Set上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  下载链接  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53(paper)   | 1088x608 |  64.4  |  55.8  | 1544  |    -   |   -   |   -   |   -   |   -   |
| DarkNet53          | 1088x608 |  64.6  |  58.5  | 1864  |  10550 | 52088 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](./jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53(paper)   | 864x480 |   62.1  |  56.9  | 1608  |    -   |   -   |   -   |   -   |   -   |
| DarkNet53          | 864x480 |   63.2  |  57.7  | 1966  |  10070 | 55081 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](./jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |   59.1  |  56.4  | 1911  |  10923 | 61789 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](./jde/jde_darknet53_30e_576x320.yml) |

**注意:**
 JDE使用8个GPU进行训练，每个GPU上batch size为4，训练了30个epoch。


### FairMOT在MOT-16 Training Set上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |  544  |  3822  | 14095 |    -     |   -   |   -   |
| DLA-34         | 1088x608 |  83.2  |  83.1  |  499  |  3861  | 14223 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  80.8  |  81.1  |  561  |  3643  | 16967 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  74.0  |  76.1  |  640  |  4989  | 23034 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_576x320.yml) |

### FairMOT在MOT-16 Test Set上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |    -   |    -   |   25.9   |    -   |   -    |
| DLA-34         | 1088x608 |  75.0  |  74.7  |  919   |  7934  |  36747 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  73.0  |  72.6  |  977   |  7578  |  40601 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  69.9  |  70.2  |  1044   |  8869  |  44898 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [配置文件](./fairmot/fairmot_dla34_30e_576x320.yml) |

**注意:**
 FairMOT DLA-34均使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


### FairMOT enhance模型
### 在MOT-16 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.9  |  74.7  |  1021   |  11425  |  31475 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |

### 在MOT-17 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.3  |  74.2  |  3270  |  29112  | 106749 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |

**注意:**
 FairMOT enhance DLA-34使用8个GPU进行训练，每个GPU上batch size为16，训练60个epoch，并且训练集中加入了crowdhuman数据集一起参与训练。


### FairMOT轻量级模型
### 在MOT-16 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  71.7  |  66.6  |  1340  |  8642  | 41592 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [配置文件](./fairmot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |

### 在MOT-17 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  70.7  |  65.7  |  4281  |  22485  | 138468 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [配置文件](./fairmot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |
| HRNetV2-W18   | 864x480  |  70.3  |  65.8  |  4056  |  18927  | 144486 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_864x480.pdparams) | [配置文件](./fairmot/fairmot_hrnetv2_w18_dlafpn_30e_864x480.yml) |
| HRNetV2-W18   | 576x320  |  65.3  |  64.8  |  4137  |  28860  | 163017 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams) | [配置文件](./fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) |

**注意:**
 FairMOT HRNetV2-W18均使用8个GPU进行训练，每个GPU上batch size为4，训练30个epoch，使用的ImageNet预训练，优化器策略采用的是Momentum，并且训练集中加入了crowdhuman数据集一起参与训练。


## 特色垂类跟踪模型

### [人头跟踪（Head Tracking)](./headtracking21/README.md)
### FairMOT在HT-21 Training Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  64.7 |  69.0  |   8533  |  148817  |  234970  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [配置文件](./headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml) |

### FairMOT在HT-21 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34         | 1088x608 |  60.8  |  62.8  |  12781   |  118109  |  198896 |    -     | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [配置文件](./headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml) |


### [行人跟踪 (Pedestrian Tracking)](./pedestrian/README.md)
### FairMOT在各个数据集val-set上Pedestrian类别的结果
|    数据集      |  输入尺寸 |  MOTA  |  IDF1  |  FPS   |  下载链接 | 配置文件 |
| :-------------| :------- | :----: | :----: | :----: | :-----: |:------: |
|  PathTrack    | 1088x608 |  44.9 |    59.3   |    -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_pathtrack.pdparams) | [配置文件](./pedestrian/fairmot_dla34_30e_1088x608_pathtrack.yml) |
|  VisDrone     | 1088x608 |  49.2 |   63.1 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_pedestrian.pdparams) | [配置文件](./pedestrian/fairmot_dla34_30e_1088x608_visdrone_pedestrian.yml) |


### [车辆跟踪 (Vehicle Tracking)](./vehicle/README.md)
### FairMOT在各个数据集val-set上Vehicle类别的结果

|    数据集      |  输入尺寸 |  MOTA  |  IDF1  |  FPS   |  下载链接 | 配置文件 |
| :-------------| :------- | :----: | :----: | :----: | :-----: |:------: |
|  BDD100K      | 1088x608 |  43.5 |  50.0  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.pdparams) | [配置文件](./vehicle/fairmot_dla34_30e_1088x608_bdd100kmot_vehicle.yml) |
|  KITTI        | 1088x608 |  82.7 |    -   |    -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_kitti_vehicle.pdparams) | [配置文件](./vehicle/fairmot_dla34_30e_1088x608_kitti_vehicle.yml) |
|  VisDrone     | 1088x608 |  52.1 |   63.3 |    -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_visdrone_vehicle.pdparams) | [配置文件](./vehicle/fairmot_dla34_30e_1088x608_visdrone_vehicle.yml) |



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

## 快速开始

### 1. 训练

FairMOT使用2个GPU通过如下命令一键式启动训练

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```

### 2. 评估

FairMOT使用单张GPU通过如下命令一键式启动评估

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```
**注意:**
 默认评估的是MOT-16 Train Set数据集，如需换评估数据集可参照以下代码修改`configs/datasets/mot.yml`，修改`data_root`：
```
EvalMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    data_root: MOT17/images/train
    keep_ori_im: False # set True if save visualization images or video
```

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4 --frame_rate=20 --save_videos
```

使用单个GPU通过如下命令预测一个图片文件夹，并保存为视频

```bash
# 预测一个图片文件夹
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --image_dir={your infer images folder} --save_videos
```

**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。`--frame_rate`表示视频的帧率，表示每秒抽取多少帧，可以自行设置，默认为-1表示会使用OpenCV读取的视频帧率。

### 4. 导出预测模型

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。

### 6. 用导出的跟踪和关键点模型Python联合预测

```bash
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```
**注意:**
 关键点模型导出教程请参考`configs/keypoint/README.md`。

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
