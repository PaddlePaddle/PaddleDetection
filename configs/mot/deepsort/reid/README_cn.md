简体中文 | [English](README.md)

# DeepSORT的ReID模型

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 由检测器和ReID模型串联组合而成，此处提供了几个常用ReID模型的配置作为DeepSORT使用的参考。

## 模型库

### 在Market1501行人重识别数据集上的结果

| 骨架网络         | 网络类型                  |  Params   |   FPS     |    mAP    |   Top1    |   Top5    |  下载链接  |   配置文件 |
| :-------------: |  :-----------------:    | :-------: |  :------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ResNet-101      |  PCB Pyramid Embedding  |  289M     |   ---     |   86.31   |   94.95   |   98.28   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)   |   [配置文件](./deepsort_pcb_pyramid_r101.yml)     |
| PPLCNet-2.5x    |  PPLCNet Embedding      |  36M      |   ---     |   71.59   |   87.38   |   95.49   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)   |   [配置文件](./deepsort_pplcnet.yml)     |

### 在VERI-Wild车辆重识别数据集上的结果

| 骨架网络         | 网络类型                  |  Params   |   FPS     |    mAP    |   Top1    |   Top5    |  下载链接  |   配置文件 |
| :-------------: |  :-----------------:    | :-------: |  :------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| PPLCNet-2.5x    |  PPLCNet Embedding      |  93M      |   ---     |   82.44   |   93.54   |   98.53   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.pdparams)   |   [配置文件](./deepsort_pplcnet_vehicle.yml)     |

**注意:**
  - ReID模型由[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供，具体训练流程和代码待PaddleClas公布.
  - 行人跟踪请用**Market1501**行人重识别数据集训练的ReID模型结合行人检测器去使用。
  - 车辆跟踪请用**VERI-Wild**车辆重识别数据集训练的ReID模型结合车辆检测器去使用。
