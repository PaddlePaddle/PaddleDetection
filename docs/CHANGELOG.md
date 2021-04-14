# 版本更新信息

## 最新版本信息

### 2.0(04.15/2021)
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
    - 新增动态图应为文档：包含首页文档，入门使用，快速开始等
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
