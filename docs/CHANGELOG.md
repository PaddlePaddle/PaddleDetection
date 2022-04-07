简体中文 | [English](./CHANGELOG_en.md)

# 版本更新信息

## 最新版本信息

### 2.4(03.24/2022)

- PP-YOLOE：
  - 发布PP-YOLOE特色模型，COCO数据集精度51.4%，V100预测速度78.1 FPS，精度速度服务器端SOTA
  - 发布s/m/l/x系列模型，打通TensorRT、ONNX部署能力
  - 支持混合精度训练，训练较PP-YOLOv2加速33%

- PP-PicoDet:
  - 发布PP-PicoDet优化模型，精度提升2%左右，CPU预测速度提升63%。
  - 新增参数量0.7M的PicoDet-XS模型
  - 后处理集成到网络中，优化端到端部署成本

- 行人分析Pipeline：
  - 发布PP-Human行人分析Pipeline，覆盖行人检测、属性识别、行人跟踪、跨镜跟踪、人流量统计、动作识别多种功能，打通TensorRT部署
  - 属性识别支持StrongBaseline模型
  - ReID支持Centroid模型
  - 动作识别支持ST-GCN摔倒检测

- 框架功能优化：
  - EMA训练速度优化20%，优化EMA训练模型保存方式
  - 支持infer预测结果保存为COCO格式

- 部署优化：
  - RCNN全系列模型支持Paddle2ONNX导出ONNX模型
  - SSD模型支持导出时融合解码OP，优化边缘端部署速度
  - 支持NMS导出TensorRT，TensorRT部署端到端速度提升

### 2.3(11.03/2021)

- 特色模型:
  - 检测: 轻量级移动端检测模型PP-PicoDet，精度速度达到移动端SOTA
  - 关键点: 轻量级移动端关键点模型PP-TinyPose

- 模型丰富度:
  - 检测：
    - 新增Swin-Transformer目标检测模型
    - 新增TOOD(Task-aligned One-stage Object Detection)模型
    - 新增GFL(Generalized Focal Loss)目标检测模型
    - 发布Sniper小目标检测优化方法，支持Faster RCNN及PP-YOLO系列模型
    - 发布针对EdgeBoard优化的PP-YOLO-EB模型

  - 跟踪
    - 发布实时跟踪系统PP-Tracking
    - 发布FairMot高精度模型、小尺度模型和轻量级模型
    - 发布行人、人头和车辆实跟踪垂类模型库，覆盖航拍监控、自动驾驶、密集人群、极小目标等场景
    - DeepSORT模型适配PP-YOLO, PP-PicoDet等更多检测器

  - 关键点
    - 新增Lite HRNet模型

- 预测部署:
  - YOLOv3系列模型支持NPU预测部署
  - FairMot模型C++预测部署打通
  - 关键点系列模型C++预测部署打通, Paddle Lite预测部署打通

- 文档:
  - 新增各系列模型英文文档

### 2.2(08.10/2021)

- 模型丰富度：
    - 发布Transformer检测模型：DETR、Deformable DETR、Sparse RCNN
    - 关键点检测新增Dark模型，发布Dark HRNet模型
    - 发布MPII数据集HRNet关键点检测模型
    - 发布人头、车辆跟踪垂类模型

- 模型优化：
    - 旋转框检测模型S2ANet发布Align Conv优化模型，DOTA数据集mAP优化至74.0

- 预测部署
    - 主流模型支持batch size>1预测部署，包含YOLOv3，PP-YOLO，Faster RCNN，SSD，TTFNet，FCOS
    - 新增多目标跟踪模型(JDE, FairMot, DeepSort) Python端预测部署支持，并支持TensorRT预测
    - 新增多目标跟踪模型FairMot联合关键点检测模型部署Python端预测部署支持
    - 新增关键点检测模型联合PP-YOLO预测部署支持

- 文档：
    - Windows预测部署文档新增TensorRT版本说明
    - FAQ文档更新发布

- 问题修复：
    - 修复PP-YOLO系列模型训练收敛性问题
    - 修复batch size>1时无标签数据训练问题


### 2.1(05.20/2021)
- 模型丰富度提升：
    - 发布关键点模型HRNet，HigherHRNet
    - 发布多目标跟踪模型DeepSort, FairMot, JDE

- 框架基础能力：
    - 支持无标注框训练

- 预测部署：
    - Paddle Inference YOLOv3系列模型支持batch size>1预测
    - 旋转框检测S2ANet模型预测部署打通
    - 增加量化模型Benchmark
    - 增加动态图模型与静态图模型Paddle-Lite demo

- 检测模型压缩：
    - 发布PPYOLO系列模型压缩模型

- 文档：
    - 更新快速开始，预测部署等教程文档
    - 新增ONNX模型导出教程
    - 新增移动端部署文档


### 2.0(04.15/2021)

  **说明：** 自2.0版本开始，动态图作为PaddleDetection默认版本，原`dygraph`目录切换为根目录，原静态图实现移动到`static`目录下。

  - 动态图模型丰富度提升：
    - 发布PP-YOLOv2及PP-YOLO tiny模型，PP-YOLOv2 COCO test数据集精度达到49.5%，V100预测速度达到68.9 FPS
    - 发布旋转框检测模型S2ANet
    - 发布两阶段实用模型PSS-Det
    - 发布人脸检测模型Blazeface

  - 新增基础模块：
    - 新增SENet，GhostNet，Res2Net骨干网络
    - 新增VisualDL训练可视化支持
    - 新增单类别精度计算及PR曲线绘制功能
    - YOLO系列模型支持NHWC数据格式

  - 预测部署：
    - 发布主要模型的预测benchmark数据
    - 适配TensorRT6，支持TensorRT动态尺寸输入，支持TensorRT int8量化预测
    - PP-YOLO, YOLOv3, SSD, TTFNet, FCOS, Faster RCNN等7类模型在Linux、Windows、NV Jetson平台下python/cpp/TRT预测部署打通:

  - 检测模型压缩：
    - 蒸馏：新增动态图蒸馏支持，并发布YOLOv3-MobileNetV1蒸馏模型
    - 联合策略：新增动态图剪裁+蒸馏联合策略压缩方案，并发布YOLOv3-MobileNetV1的剪裁+蒸馏压缩模型
    - 问题修复：修复动态图量化模型导出问题

  - 文档：
    - 新增动态图英文文档：包含首页文档，入门使用，快速开始，模型算法、新增数据集等
    - 新增动态图中英文安装文档
    - 新增动态图RCNN系列和YOLO系列配置文件模板及配置项说明文档


## 历史版本信息

### 2.0-rc(02.23/2021)
  - 动态图模型丰富度提升：
    - 优化RCNN模型组网及训练方式，RCNN系列模型精度提升(依赖Paddle develop或2.0.1版本)
    - 新增支持SSDLite，FCOS，TTFNet，SOLOv2系列模型
    - 新增行人和车辆垂类目标检测模型

  - 新增动态图基础模块：
    - 新增MobileNetV3，HRNet骨干网络
    - 优化RoIAlign计算逻辑，RCNN系列模型精度提升(依赖Paddle develop或2.0.1版本)
    - 新增支持Synchronized Batch Norm
    - 新增支持Modulated Deformable Convolution

  - 预测部署：
    - 发布动态图python、C++、Serving部署解决方案及文档，支持Faster RCNN，Mask RCNN，YOLOv3，PP-YOLO，SSD，TTFNet，FCOS，SOLOv2等系列模型预测部署
    - 动态图预测部署支持TensorRT模式FP32，FP16推理加速

  - 检测模型压缩：
    - 裁剪：新增动态图裁剪支持，并发布YOLOv3-MobileNetV1裁剪模型
    - 量化：新增动态图量化支持，并发布YOLOv3-MobileNetV1和YOLOv3-MobileNetV3量化模型

  - 文档：
    - 新增动态图入门教程文档：包含安装说明，快速开始，准备数据，训练/评估/预测流程文档
    - 新增动态图进阶教程文档：包含模型压缩、推理部署文档
    - 新增动态图模型库文档

### v2.0-beta(12.20/2020)
  - 动态图支持:
    - 支持Faster-RCNN, Mask-RCNN, FPN, Cascade Faster/Mask RCNN, YOLOv3和SSD模型，试用版本。
  - 模型提升：
    - 更新PP-YOLO MobileNetv3 large和small模型，精度提升，并新增裁剪和蒸馏后的模型。
  - 新功能：
    - 支持VisualDL可视化数据预处理图片。

  - Bug修复:
    - 修复BlazeFace人脸关键点预测bug。


### v0.5.0(11/2020)
  - 模型丰富度提升：
    - 发布SOLOv2系列模型，其中SOLOv2-Light-R50-VD-DCN-FPN 模型在单卡V100上达到 38.6 FPS，加速24% ，COCO验证集精度达到38.8%, 提升2.4绝对百分点。
    - 新增Android移动端检测demo，包括SSD、YOLO系列模型，可直接扫码安装体验。

  - 移动端模型优化：
    - 新增PACT新量化策略，YOLOv3-Mobilenetv3在COCO数据集上比普通量化相比提升0.7%。

  - 易用性提升及功能组件：
    - 增强generate_proposal_labels算子功能，规避模型出nan风险。
    - 修复deploy下python与C++预测若干问题。
    - 统一COCO与VOC数据集下评估流程，支持输出单类AP和P-R曲线。
    - PP-YOLO支持矩形输入图像。

  - 文档：
    - 新增目标检测全流程教程，新增Jetson平台部署教程。


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

  - 易用性提升及功能组件：
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
    - 文档部署到网站: https://paddledetection.readthedocs.io

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
