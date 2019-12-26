[English](README_en.md) | 简体中文

# PaddleDetection

PaddleDetection的目的是为工业界和学术界提供丰富、易用的目标检测模型。不仅性能优越、易于部署，而且能够灵活的满足算法研究的需求。

**目前检测库下模型均要求使用PaddlePaddle 1.6及以上版本或适当的develop版本。**

<div align="center">
  <img src="demo/output/000000570688.jpg" />
</div>


## 简介

特性：

- 易部署:

  PaddleDetection的模型中使用的核心算子均通过C++或CUDA实现，同时基于PaddlePaddle的高性能推理引擎可以方便地部署在多种硬件平台上。

- 高灵活度：

  PaddleDetection通过模块化设计来解耦各个组件，基于配置文件可以轻松地搭建各种检测模型。

- 高性能：

  基于PaddlePaddle框架的高性能内核，在模型训练速度、显存占用上有一定的优势。例如，YOLOv3的训练速度快于其他框架，在Tesla V100 16GB环境下，Mask-RCNN(ResNet50)可以单卡Batch Size可以达到4 (甚至到5)。

支持的模型结构：

|                    | ResNet | ResNet-vd <sup>[1](#vd)</sup> | ResNeXt-vd | SENet | MobileNet | DarkNet | VGG | HRNet | Res2Net |
|--------------------|:------:|------------------------------:|:----------:|:-----:|:---------:|:-------:|:---:|:-----:| :--:    |
| Faster R-CNN       | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   | ✗    |  ✗       |
| Faster R-CNN + FPN | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   | ✓    |  ✓       |
| Mask R-CNN         | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   | ✗    |  ✗       |
| Mask R-CNN + FPN   | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   | ✗    |  ✓       |
| Cascade Faster-RCNN | ✓      |                             ✓ | ✓          | ✗     | ✗         | ✗       | ✗  | ✗    |  ✗       |
| Cascade Mask-RCNN  | ✓      |                             ✗ | ✗          | ✓     | ✗         | ✗       | ✗   | ✗    |  ✗       |
| RetinaNet          | ✓      |                             ✗ | ✓          | ✗     | ✗         | ✗       | ✗   | ✗    |  ✗       |
| YOLOv3             | ✓      |                             ✗ | ✗          | ✗     | ✓         | ✓       | ✗   | ✗    |  ✗       |
| SSD                | ✗      |                             ✗ | ✗          | ✗     | ✓         | ✗       | ✓   | ✗    |  ✗       |

<a name="vd">[1]</a> [ResNet-vd](https://arxiv.org/pdf/1812.01187) 模型提供了较大的精度提高和较少的性能损失。

扩展特性：

- [x] **Synchronized Batch Norm**: 目前在YOLOv3中使用。
- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Deformable PSRoI Pooling**

**注意:** Synchronized batch normalization 只能在多GPU环境下使用，不能在CPU环境或者单GPU环境下使用。


## 使用教程

- [安装说明](docs/INSTALL_cn.md)
- [快速开始](docs/QUICK_STARTED_cn.md)
- [训练、评估流程](docs/GETTING_STARTED_cn.md)
- [数据预处理及自定义数据集](docs/DATA_cn.md)
- [配置模块设计和介绍](docs/CONFIG_cn.md)
- [详细的配置信息和参数说明示例](docs/config_example/)
- [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)
- [迁移学习教程](docs/TRANSFER_LEARNING_cn.md)

## 模型库

- [模型库](docs/MODEL_ZOO_cn.md)
- [人脸检测模型](configs/face_detection/README.md)
- [行人检测和车辆检测预训练模型](contrib/README_cn.md) 针对不同场景的检测模型
- [YOLOv3增强模型](docs/YOLOv3_ENHANCEMENT.md) 改进原始YOLOv3，精度达到41.4%，原论文精度为33.0%，同时预测速度也得到提升
- [Objects365 2019 Challenge夺冠模型](docs/CACascadeRCNN.md) Objects365 Full Track任务中最好的单模型之一,精度达到31.7%
- [Open Images V5和Objects365数据集模型](docs/OIDV5_BASELINE_MODEL.md)


## 模型压缩
- [量化训练压缩示例](slim/quantization)
- [剪枝压缩示例](slim/prune)

## 推理部署

- [模型导出教程](docs/EXPORT_MODEL.md)
- [C++推理部署](inference/README.md)

## Benchmark

- [推理Benchmark](docs/BENCHMARK_INFER_cn.md)



## 版本更新

### 12/2019
- 增加Res2Net模型。
- 增加HRNet模型。
- 增加GIOU loss和DIOU loss。


### 21/11/2019
- 增加CascadeClsAware RCNN模型。
- 增加CBNet，ResNet200和Non-local模型。
- 增加SoftNMS。
- 增加Open Image V5数据集和Objects365数据集模型。

### 10/2019
- 增加增强版YOLOv3模型，精度高达41.4%。
- 增加人脸检测模型BlazeFace、Faceboxes。
- 丰富基于COCO的模型，精度高达51.9%。
- 增加Objects365 2019 Challenge上夺冠的最佳单模型之一CACascade-RCNN。
- 增加行人检测和车辆检测预训练模型。
- 支持FP16训练。
- 增加跨平台的C++推理部署方案。
- 增加模型压缩示例。


### 2/9/2019
- 增加GroupNorm模型。
- 增加CascadeRCNN+Mask模型。

#### 5/8/2019
- 增加Modulated Deformable Convolution系列模型。

#### 29/7/2019

- 增加检测库中文文档
- 修复R-CNN系列模型训练同时进行评估的问题
- 新增ResNext101-vd + Mask R-CNN + FPN模型
- 新增基于VOC数据集的YOLOv3模型

#### 3/7/2019

- 首次发布PaddleDetection检测库和检测模型库
- 模型包括：Faster R-CNN, Mask R-CNN, Faster R-CNN+FPN, Mask
  R-CNN+FPN, Cascade-Faster-RCNN+FPN, RetinaNet, YOLOv3, 和SSD.

## 如何贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。
