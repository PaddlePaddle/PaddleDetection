[English](README_en.md) | 简体中文

# PaddleDetection

PaddleDetection的目的是为工业界和学术界提供丰富、易用的目标检测模型。不仅性能优越、易于部署，而且能够灵活的满足算法研究的需求。

**目前检测库下模型均要求使用PaddlePaddle 1.7及以上版本或适当的develop版本。**

<div align="center">
  <img src="docs/images/000000570688.jpg" />
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

|                    | ResNet | ResNet-vd <sup>[1](#vd)</sup> | ResNeXt-vd | SENet | MobileNet |  HRNet | Res2Net |
|--------------------|:------:|------------------------------:|:----------:|:-----:|:---------:|:------:| :--:    |
| Faster R-CNN       | ✓      |                             ✓ | x          | ✓     | ✗         |  ✗     |  ✗      |
| Faster R-CNN + FPN | ✓      |                             ✓ | ✓          | ✓     | ✗         |  ✓     |  ✓      |
| Mask R-CNN         | ✓      |                             ✓ | x          | ✓     | ✗         |  ✗     |  ✗      |
| Mask R-CNN + FPN   | ✓      |                             ✓ | ✓          | ✓     | ✗         |  ✗     |  ✓      |
| Cascade Faster-RCNN | ✓     |                             ✓ | ✓          | ✗     | ✗         |  ✗     |  ✗      |
| Cascade Mask-RCNN  | ✓      |                             ✗ | ✗          | ✓     | ✗         |  ✗     |  ✗      |
| Libra R-CNN        | ✗      |                             ✓ | ✗          | ✗     | ✗         |  ✗     |  ✗      |
| RetinaNet          | ✓      |                             ✗ | ✓          | ✗     | ✗         |  ✗     |  ✗      |
| YOLOv3             | ✓      |                             ✗ | ✗          | ✗     | ✓         |  ✗     |  ✗      |
| SSD                | ✗      |                             ✗ | ✗          | ✗     | ✓         |  ✗     |  ✗      |
| BlazeFace          | ✗      |                             ✗ | ✗          | ✗     | ✗         |  ✗     |  ✗      |
| Faceboxes          | ✗      |                             ✗ | ✗          | ✗     | ✗         |  ✗     |  ✗      |

<a name="vd">[1]</a> [ResNet-vd](https://arxiv.org/pdf/1812.01187) 模型提供了较大的精度提高和较少的性能损失。

更多的Backone：

- DarkNet
- VGG
- GCNet
- CBNet

扩展特性：

- [x] **Synchronized Batch Norm**: 目前在YOLOv3中使用。
- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Deformable PSRoI Pooling**
- [x] **Non-local和GCNet**

**注意:** Synchronized batch normalization 只能在多GPU环境下使用，不能在CPU环境或者单GPU环境下使用。

## 文档教程

**最新动态：** 已发布文档教程：[https://paddledetection.readthedocs.io](https://paddledetection.readthedocs.io)

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [快速开始](docs/tutorials/QUICK_STARTED_cn.md)
- [训练/评估/预测流程](docs/tutorials/GETTING_STARTED_cn.md)

### 进阶教程
- [数据预处理及自定义数据集](docs/advanced_tutorials/READER.md)
- [搭建模型步骤](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- [配置模块设计和介绍](docs/advanced_tutorials/CONFIG_cn.md)
- [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)
- [迁移学习教程](docs/advanced_tutorials/TRANSFER_LEARNING_cn.md)
- [模型压缩](slim)
    - [压缩benchmark](slim)
    - [量化](slim/quantization)
    - [剪枝](slim/prune)
    - [蒸馏](slim/distillation)
    - [神经网络搜索](slim/nas)
- [推理部署](inference)
    - [模型导出教程](docs/advanced_tutorials/inference/EXPORT_MODEL.md)
    - [预测引擎Python API使用示例](docs/advanced_tutorials/inference/INFERENCE.md)
    - [C++推理部署](inference/README.md)
    - [推理Benchmark](docs/advanced_tutorials/inference/BENCHMARK_INFER_cn.md)

## 模型库

- [模型库](docs/MODEL_ZOO_cn.md)
- [人脸检测模型](configs/face_detection/README.md)
- [行人检测和车辆检测预训练模型](contrib/README_cn.md) 针对不同场景的检测模型
- [YOLOv3增强模型](docs/featured_model/YOLOv3_ENHANCEMENT.md) 改进原始YOLOv3，精度达到41.4%，原论文精度为33.0%，同时预测速度也得到提升
- [Objects365 2019 Challenge夺冠模型](docs/featured_model/CACascadeRCNN.md) Objects365 Full Track任务中最好的单模型之一,精度达到31.7%
- [Open Images V5和Objects365数据集模型](docs/featured_model/OIDV5_BASELINE_MODEL.md)


## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

## 版本更新
v0.2.0版本已经在`02/2020`发布，增加多个模型，升级数据处理模块，拆分YOLOv3的loss，修复已知诸多bug等，
详细内容请参考[版本更新文档](docs/CHANGELOG.md)。

## 如何贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。
