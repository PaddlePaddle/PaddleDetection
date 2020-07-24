# 版本更新信息

## 最新版本信息

### v0.4.0(07/2020)
  - 模型丰富度提升：
    - 发布PPYOLO模型，COCO数据集精度达到45.2%，单卡V100预测速度达到72.9 FPS，精度和预测速度优于YOLOv4模型。
    - 新增TTFNet模型，base版本对齐竞品，COCO数据集精度达到32.9%。
    - 新增HTC模型，base版本对齐竞品，COCO数据集精度达到42.2%。
    - 新增BlazeFace人脸关键点检测模型，在Wider-Face数据集的Easy-Set精度达到85.2%。
    - 新增ACFPN模型， COCO数据集精度达到39.6%。
    - 发布服务器端通用目标检测模型（包含676类），相同策略在COCO数据集上，V100为19.5FPS时，COCO mAP可以达到49.4%。

  - 移动端模型优化：
    - 新增SSDLite系列优化模型，包括新增GhostNet的Backbone，新增FPN组件等，精度提升0.5%-1.5%。

  - 易用性提升机功能组件：
    - 新增GridMask, RandomErasing数据增强方法。
    - 新增Matrix NMS支持。
    - 新增EMA(Exponential Moving Average)训练支持。
    - 新增多机训练方法，两机相对于单机平均加速比80%，多机训练支持待进一步验证。

### v0.3.0(05/2020)
  - 模型丰富度提升：
    - 添加Efficientdet-D0模型，速度与精度优于竞品。
    - 新增YOLOv4预测模型，精度对齐竞品；新增YOLOv4在Pascal VOC数据集上微调训练，精度达到85.5%。
    - YOLOv3新增MobileNetV3骨干网络，COCO数据集精度达到31.6%。
    - 添加Anchor-free模型FCOS，精度优于竞品。
    - 添加Anchor-free模型CornernetSqueeze，精度优于竞品，优化模型的COCO数据集精度38.2%, +3.7%，速度较YOLOv3-Darknet53快5%。
    - 添加服务器端实用目标检测模型CascadeRCNN-ResNet50vd模型，速度与精度优于竞品EfficientDet。

  - 移动端推出3种模型：
    - SSDLite系列模型：SSDLite-Mobilenetv3 small/large模型，精度优于竞品。
    - YOLOv3移动端方案: YOLOv3-MobileNetv3模型压缩后加速3.5倍，速度和精度均领先于竞品的SSDLite模型。
    - RCNN移动端方案：CascadeRCNN-MobileNetv3经过系列优化, 推出输入图像分别为320x320和640x640的模型，速度与精度具有较高性价比。

  - 预测部署重构：
    - 新增Python预测部署流程，支持RCNN，YOLO，SSD，RetinaNet，人脸系列模型，支持视频预测。
    - 重构C++预测部署，提高易用性。

  - 易用性提升及功能组件：
    - 增加AutoAugment数据增强。
    - 升级检测库文档结构。
    - 支持迁移学习自动进行shape匹配。
    - 优化mask分支评估阶段内存占用。


## 历史版本信息

### v0.2.0(02/2020)
  - 新增模型:
    - 新增基于CBResNet模型。
    - 新增LibraRCNN模型。
    - 进一步提升YOLOv3模型精度，基于COCO数据精度达到43.2%，相比上个版本提升1.4%。
  - 新增基础模块:
    - 主干网络: 新增CBResNet。
    - loss模块: YOLOv3的loss支持细粒度op组合。
    - 正则模块: 新增DropBlock模块。
  - 功能优化和改进:
    - 加速YOLOv3数据预处理，整体训练提速40%。
    - 优化数据预处理逻辑，提升易用性。
    - 增加人脸检测预测benchmark数据。
    - 增加C++预测引擎Python API预测示例。
  - 检测模型压缩 :
    - 裁剪: 发布MobileNet-YOLOv3裁剪方案和模型，基于VOC数据FLOPs - 69.6%, mAP + 1.4%，基于COCO数据FLOPS-28.8%, mAP + 0.9%; 发布ResNet50vd-dcn-YOLOv3裁剪方案和模型，基于COCO数据集FLOPS - 18.4%, mAP + 0.8%。
    - 蒸馏: 发布MobileNet-YOLOv3蒸馏方案和模型，基于VOC数据mAP + 2.8%，基于COCO数据mAP + 2.1%。
    - 量化: 发布YOLOv3-MobileNet和BlazeFace的量化模型。
    - 裁剪+蒸馏: 发布MobileNet-YOLOv3裁剪+蒸馏方案和模型，基于COCO数据FLOPS - 69.6%，基于TensorRT预测加速64.5%，mAP - 0.3 %; 发布ResNet50vd-dcn-YOLOv3裁剪+蒸馏方案和模型，基于COCO数据FLOPS - 43.7%，基于TensorRT预测加速24.0%，mAP + 0.6 %。
    - 搜索: 开源BlazeFace-Nas的完成搜索方案。
  - 预测部署:
    - 集成 TensorRT，支持FP16、FP32、INT8量化推理加速。
  - 文档:
    - 增加详细的数据预处理模块介绍文档以及实现自定义数据Reader文档。
    - 增加如何新增算法模型的文档。
    - 文档部署到网站: https://paddledetection.readthedocs.io/zh/latest/

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

### 5/8/2019
- 增加Modulated Deformable Convolution系列模型。

### 29/7/2019

- 增加检测库中文文档
- 修复R-CNN系列模型训练同时进行评估的问题
- 新增ResNext101-vd + Mask R-CNN + FPN模型
- 新增基于VOC数据集的YOLOv3模型

### 3/7/2019

- 首次发布PaddleDetection检测库和检测模型库
- 模型包括：Faster R-CNN, Mask R-CNN, Faster R-CNN+FPN, Mask
  R-CNN+FPN, Cascade-Faster-RCNN+FPN, RetinaNet, YOLOv3, 和SSD.
