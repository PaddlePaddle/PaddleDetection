# YOLOv3增强模型

---
## 简介

[YOLOv3](https://arxiv.org/abs/1804.02767) 是由 [Joseph Redmon](https://arxiv.org/search/cs?searchtype=author&query=Redmon%2C+J) 和 [Ali Farhadi](https://arxiv.org/search/cs?searchtype=author&query=Farhadi%2C+A) 提出的单阶段检测器, 该检测
器与达到同样精度的传统目标检测方法相比，推断速度能达到接近两倍.

PaddleDetection实现版本中使用了 [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3) 中提出的图像增强和label smooth等优化方法，精度优于darknet框架的实现版本，在COCO-2017数据集上，YOLOv3(DarkNet)达到`mAP(0.50:0.95)= 38.9`的精度，比darknet实现版本的精度(33.0)要高5.9。同时，在推断速度方面，基于Paddle预测库的加速方法，推断速度比darknet高30%。

在此基础上，PaddleDetection对YOLOv3进一步改进，得到了更大的精度和速度优势。


## 方法描述

将YOLOv3骨架网络更换为ResNet50-vd，同时在最后一个Residual block中引入[Deformable convolution v2](https://arxiv.org/abs/1811.11168)(可变形卷积)替代原始卷积操作。另外，使用[object365数据集](https://www.objects365.org/download.html)训练得到的模型作为coco数据集上的预训练模型，进一步提高YOLOv3的精度。

## 使用方法

### 模型训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -c configs/dcn/yolov3_r50vd_dcn.yml
```

更多模型参数请使用``python tools/train.py --help``查看，或参考[训练、评估及参数说明](docs/GETTING_STARTED_cn.md)文档

### 模型效果

|           模型         |     预训练模型     |    验证集 mAP   |        P4预测速度       |                         下载                           |
| :---------------------:|:-----------------: | :-------------: | :----------------------:|:-----------------------------------------------------: |
|      YOLOv3 DarkNet    |  [DarkNet pretrain](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_pretrained.tar)             | 38.9 | 原生：88.3ms<br>tensorRT-FP32: 42.5ms | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| YOLOv3 ResNet50_vd dcn | [ImageNet pretrain](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar)           | 39.1 | 原生：74.4ms<br>tensorRT-FP32: 35.2ms | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_imagenet.tar) |
| YOLOv3 ResNet50_vd dcn | [Object365 pretrain](https://paddlemodels.bj.bcebos.com/object_detection/ResNet50_vd_obj365_pretrained.tar) | 41.4 | 原生：74.4ms<br>tensorRT-FP32: 35.2ms | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365.tar) |
