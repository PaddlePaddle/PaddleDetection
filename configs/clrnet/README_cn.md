简体中文 | [English](README.md)

# CLRNet (CLRNet: Cross Layer Refinement Network for Lane Detection)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [引用](#引用)

## 内容

[CLRNet](https://arxiv.org/abs/2203.10350)是一个车道线检测模型。CLRNet模型设计了车道线检测的直线先验轨迹，车道线iou以及nms方法，融合提取车道线轨迹的上下文高层特征与底层特征，利用FPN多尺度进行refine，在车道线检测相关数据集取得了SOTA的性能。

## 模型库

### CLRNet在CUlane上结果

| 骨架网络       | mF1 | F1@50   |    F1@75    | 下载链接 | 配置文件 |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| ResNet-18         | 55.39 |  79.56  |    62.83   | [下载链接]() | [配置文件](./clr_resnet18_culane.yml) |


## 引用
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```
