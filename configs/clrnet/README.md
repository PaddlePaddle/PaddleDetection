English | [简体中文](README_cn.md)

# CLRNet (CLRNet: Cross Layer Refinement Network for Lane Detection)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Citations](#Citations)

## Introduction

[CLRNet](https://arxiv.org/abs/2203.10350)is a lane detection model. The CLRNet model is designed with line prior for lane detection, line iou loss as well as nms method, fused to extract contextual high-level features of lane line with low-level features, and refined by FPN multi-scale.Finally, the model achieved SOTA performance in lane detection datasets.

## Model Zoo

### CenterNet Results on COCO-val 2017

| backbone       | mF1 | F1@50   |    F1@75    | download | config |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| ResNet-18         | 55.39 |  79.56  |    62.83   | [model]() | [config](./clr_resnet18_culane.yml) |


## Citations
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
