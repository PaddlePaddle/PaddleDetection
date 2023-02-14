# 模型库和基线

# 内容
- [基础设置](#基础设置)
    - [测试环境](#测试环境)
    - [通用设置](#通用设置)
    - [训练策略](#训练策略)
    - [ImageNet预训练模型](#ImageNet预训练模型)
- [基线](#基线)
    - [目标检测](#目标检测)
    - [实例分割](#实例分割)
    - [PaddleYOLO](#PaddleYOLO)
    - [人脸检测](#人脸检测)
    - [旋转框检测](#旋转框检测)
    - [关键点检测](#关键点检测)
    - [多目标跟踪](#多目标跟踪)

# 基础设置

## 测试环境

- Python 3.7
- PaddlePaddle 每日版本
- CUDA 10.1
- cuDNN 7.5
- NCCL 2.4.8

## 通用设置

- 所有模型均在COCO17数据集中训练和测试。
- [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov5)、[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov6)、[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov7)和[YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov8)这几类模型的代码在[PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)中，**PaddleYOLO库开源协议为GPL 3.0**。
- 除非特殊说明，所有ResNet骨干网络采用[ResNet-B](https://arxiv.org/pdf/1812.01187)结构。
- **推理时间(fps)**: 推理时间是在一张Tesla V100的GPU上通过'tools/eval.py'测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

## 训练策略

- 我们采用和[Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules)相同的训练策略。
- 1x 策略表示：在总batch size为8时，初始学习率为0.01，在8 epoch和11 epoch后学习率分别下降10倍，最终训练12 epoch。
- 2x 策略为1x策略的两倍，同时学习率调整的epoch数位置也为1x的两倍。

## ImageNet预训练模型

Paddle提供基于ImageNet的骨架网络预训练模型。所有预训练模型均通过标准的Imagenet-1k数据集训练得到，ResNet和MobileNet等是采用余弦学习率调整策略或SSLD知识蒸馏训练得到的高精度预训练模型，可在[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)查看模型细节。


# 基线

## 目标检测

### Faster R-CNN

请参考[Faster R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/faster_rcnn/)

### YOLOv3

请参考[YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/)

### PP-YOLOE/PP-YOLOE+

请参考[PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe/)

### PP-YOLO/PP-YOLOv2

请参考[PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/)

### PicoDet

请参考[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet)

### RetinaNet

请参考[RetinaNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/retinanet/)

### Cascade R-CNN

请参考[Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/cascade_rcnn)

### SSD/SSDLite

请参考[SSD](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ssd/)

### FCOS

请参考[FCOS](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/fcos/)

### CenterNet

请参考[CenterNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/centernet/)

### TTFNet/PAFNet

请参考[TTFNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/)

### Group Normalization

请参考[Group Normalization](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gn/)

### Deformable ConvNets v2

请参考[Deformable ConvNets v2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/dcn/)

### HRNets

请参考[HRNets](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/hrnet/)

### Res2Net

请参考[Res2Net](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/res2net/)

### ConvNeXt

请参考[ConvNeXt](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/convnext/)

### GFL

请参考[GFL](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl)

### TOOD

请参考[TOOD](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/tood)

### PSS-DET(RCNN-Enhance)

请参考[PSS-DET](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rcnn_enhance)

### DETR

请参考[DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/detr)

### Deformable DETR

请参考[Deformable DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/deformable_detr)

### Sparse R-CNN

请参考[Sparse R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/sparse_rcnn)

###  Vision Transformer

请参考[Vision Transformer](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/vitdet)

### DINO

请参考[DINO](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/dino)

### YOLOX

请参考[YOLOX](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolox)

### YOLOF

请参考[YOLOF](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolof)


## 实例分割

### Mask R-CNN

请参考[Mask R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mask_rcnn/)

### Cascade R-CNN

请参考[Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/cascade_rcnn)

### SOLOv2

请参考[SOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/solov2/)

### QueryInst

请参考[QueryInst](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/queryinst)


## [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)

请参考[PaddleYOLO模型库](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/docs/MODEL_ZOO_cn.md)

### YOLOv5

请参考[YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov5)

### YOLOv6(v3.0)

请参考[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov6)

### YOLOv7

请参考[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov7)

### YOLOv8

请参考[YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov8)

### RTMDet

请参考[RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/rtmdet)


## 人脸检测

请参考[人脸检测模型库](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/face_detection)

### BlazeFace

请参考[BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/face_detection/)


## 旋转框检测

请参考[旋转框检测模型库](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate)

### PP-YOLOE-R

请参考[PP-YOLOE-R](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r)

### FCOSR

请参考[FCOSR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr)

### S2ANet

请参考[S2ANet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet)


## 关键点检测

请参考[关键点检测模型库](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint)

### PP-TinyPose

请参考[PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/tiny_pose)

### HRNet

请参考[HRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/hrnet)

### Lite-HRNet

请参考[Lite-HRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/lite_hrnet)

### HigherHRNet

请参考[HigherHRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/higherhrnet)


## 多目标跟踪

请参考[多目标跟踪模型库](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot)

### DeepSORT

请参考[DeepSORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/deepsort)

### ByteTrack

请参考[ByteTrack](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/bytetrack)

### OC-SORT

请参考[OC-SORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/ocsort)

### BoT-SORT

请参考[BoT-SORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/botsort)

### CenterTrack

请参考[CenterTrack](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/centertrack)

### FairMOT/MC-FairMOT

请参考[FairMOT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/fairmot)

### JDE

请参考[JDE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde)
