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

Paddle提供基于ImageNet的骨架网络预训练模型。所有预训练模型均通过标准的Imagenet-1k数据集训练得到。[下载链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances)

- 注：ResNet50模型通过余弦学习率调整策略训练得到。[ResNet50下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar),
 [ResNet50_vd下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar)

## 基线

### Faster & Mask R-CNN

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     ----     |  35.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn_r50_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   1x    |     ----     |  37.0  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50             | Mask         |    1    |   1x    |     ----     |  36.4  |    31.9    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/mask_rcnn_r50_1x_coco.yml) |
| ResNet50-FPN         | Mask         |    1    |   1x    |     ----     |  38.3  |    34.5    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/mask_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Cascade Faster         |    1    |   1x    |     ----     |  41.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/cascade_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/cascade_faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Cascade Mask         |    1    |   1x    |     ----     |  41.6  |    35.3    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/cascade_mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/cascade_mask_rcnn_r50_fpn_1x_coco.yml) |
| DarkNet53         | YOLOv3         |    1    |   270e    |     ----     |  39.0  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/yolov3_darknet53_270e_coco.yml) |
