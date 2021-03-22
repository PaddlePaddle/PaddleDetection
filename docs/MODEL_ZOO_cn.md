# 模型库和基线

## 测试环境

- Python 3.7
- PaddlePaddle 每日版本
- CUDA 9.0
- cuDNN >=7.4
- NCCL 2.1.2

## 通用设置

- 所有模型均在COCO17数据集中训练和测试。
- 除非特殊说明，所有ResNet骨干网络采用[ResNet-B](https://arxiv.org/pdf/1812.01187)结构。
- 对于RCNN和RetinaNet系列模型，训练阶段仅使用水平翻转作为数据增强，测试阶段不使用数据增强。
- **推理时间(fps)**: 推理时间是在一张Tesla V100的GPU上通过'tools/eval.py'测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

## 训练策略

- 我们采用和[Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules)相同的训练策略。
- 1x 策略表示：在总batch size为8时，初始学习率为0.01，在8 epoch和11 epoch后学习率分别下降10倍，最终训练12 epoch。
- 2x 策略为1x策略的两倍，同时学习率调整位置也为1x的两倍。

## ImageNet预训练模型

Paddle提供基于ImageNet的骨架网络预训练模型。所有预训练模型均通过标准的Imagenet-1k数据集训练得到，ResNet和MobileNet等是采用余弦学习率调整策略或SSLD知识蒸馏训练得到的高精度预训练模型，可在[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)查看模型细节。


## 基线

### Faster R-CNN

请参考[Faster R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/)

### Mask R-CNN

请参考[Mask R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/)

### Cascade R-CNN

请参考[Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/cascade_rcnn/)

### YOLOv3

请参考[YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/)

### SSD

请参考[SSD](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ssd/)

### FCOS

请参考[FCOS](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/fcos/)

### SOLOv2

请参考[SOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/solov2/)

### PP-YOLO

请参考[PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/)

### TTFNet

请参考[TTFNet](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ttfnet/)
