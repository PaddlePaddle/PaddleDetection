# 模型库和基线

## 测试环境

- Python 2.7.1
- PaddlePaddle >=1.5
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
- 1x 策略表示：在总batch size为16时，初始学习率为0.02，在6万轮和8万轮后学习率分别下降10倍，最终训练9万轮。在总batch size为8时，初始学习率为0.01，在12万轮和16万轮后学习率分别下降10倍，最终训练18万轮。
- 2x 策略为1x策略的两倍，同时学习率调整位置也为1x的两倍。

## ImageNet预训练模型

Paddle提供基于ImageNet的骨架网络预训练模型。所有预训练模型均通过标准的Imagenet-1k数据集训练得到。[下载链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances)

- 注：ResNet50模型通过余弦学习率调整策略训练得到。[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar)

## 基线

### Faster & Mask R-CNN

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: |
| ResNet50             | Faster         |    1    |   1x    |     12.747     |  35.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) |
| ResNet50             | Faster         |    1    |   2x    |     12.686     |  37.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) |
| ResNet50             | Mask           |    1    |   1x    |     11.615     |  36.5  |  32.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) |
| ResNet50             | Mask           |    1    |   2x    |     11.494     |  38.2  |  33.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) |
| ResNet50-vd          | Faster         |    1    |   1x    |     12.575     |  36.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   1x    |     22.273     |  37.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   2x    |     22.297     |  37.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Mask           |    1    |   1x    |     15.184     |  37.9  |  34.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Mask           |    1    |   2x    |     15.881     |  38.7  |  34.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |     17.507     |  40.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Cascade Mask   |    1    |   1x    |       -        |  41.3  |  35.5   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Faster         |    2    |   2x    |     21.847     |  38.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) |
| ResNet50-vd-FPN      | Mask           |    1    |   2x    |     15.825     |  39.8  |  35.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) |
| ResNet101            | Faster         |    1    |   1x    |     9.316      |  38.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   1x    |     17.297     |  38.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   2x    |     17.246     |  39.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| ResNet101-FPN        | Mask           |    1    |   1x    |     12.983     |  39.5  |  35.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |     17.011     |  40.5  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |     16.934     |  40.8  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Mask           |    1    |   1x    |     13.105     |  41.4  |  36.8   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |     8.815      |  42.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |     8.809      |  41.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   1x    |     7.689      |  42.9  |  37.9   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   2x    |     7.859      |  42.6  |  37.6   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| SENet154-vd-FPN      | Faster         |    1    |  1.44x  |     3.408      |  42.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_vd_fpn_s1x.tar) |
| SENet154-vd-FPN      | Mask           |    1    |  1.44x  |     3.233      |  44.0  |  38.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |

### Deformable 卷积网络v2

| 骨架网络             | 网络类型           | 卷积    | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | Mask AP |                           下载                           |
| :------------------- | :------------- | :-----: |:--------: | :-----: | :-----------: |:----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN         | Faster         | c3-c5   |    2      |   1x    |    19.978     |  41.0  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Faster         | c3-c5   |    2      |   2x    |    19.222     |  42.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Faster         | c3-c5   |    2      |   1x    |    14.477     |  44.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         | c3-c5   |    1      |   1x    |    7.209      |  45.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN         | Mask           | c3-c5   |    1      |   1x    |    14.53      |  41.9  |  37.3   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Mask           | c3-c5   |    1      |   2x    |    14.832     |  42.9  |  38.0   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Mask           | c3-c5   |    1      |   1x    |    11.546     |  44.6  |  39.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Mask           | c3-c5   |    1      |   1x    |     6.45      |  46.2  |  40.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN         | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  44.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet101-vd-FPN     | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  46.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  47.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| SENet154-vd-FPN      | Cascade Mask   | c3-c5   |    1      |  1.44x  |      -        |  51.9  |  43.9   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_dcnv2_se154_vd_fpn_gn_s1x.tar) |

#### 注意事项:
- Deformable卷积网络v2(dcn_v2)参考自论文[Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5`意思是在resnet模块的3到5阶段增加`dcn`.
- 详细的配置文件在[configs/dcn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection/configs/dcn)

### Group Normalization
| 骨架网络             | 网络类型           | 每张GPU图片个数 | 学习率策略 | Box AP | Mask AP |                           下载                           |
| :------------------- | :------------- |:--------: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN         | Faster         |    2      |   2x    |  39.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_gn_2x.tar) |
| ResNet50-FPN         | Mask           |    1      |   2x    |  40.1  |   35.8  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_gn_2x.tar) |

#### 注意事项:
- Group Normalization参考论文[Group Normalization](https://arxiv.org/abs/1803.08494).
- 详细的配置文件在[configs/gn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection/configs/gn)

### Yolo v3

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |    45.571     |  38.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 416  |    8    |   270e  |      -        |  37.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 320  |    8    |   270e  |      -        |  34.8  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |    78.302     |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |      -        |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |      -        |  27.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| ResNet34     | 608  |    8    |   270e  |    63.356     |  36.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 416  |    8    |   270e  |      -        |  34.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 320  |    8    |   270e  |      -        |  31.4  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |

### Yolo v3 基于Pasacl VOC数据集

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |    54.977     |  83.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 416  |    8    |   270e  |      -        |  83.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 320  |    8    |   270e  |      -        |  82.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |   104.291     |  76.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |      -        |  76.7  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |      -        |  75.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| ResNet34     | 608  |    8    |   270e  |    82.247     |  82.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 416  |    8    |   270e  |      -        |  81.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 320  |    8    |   270e  |      -        |  80.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |

**注意事项:** Yolo v3在8卡，总batch size为64下训练270轮。数据增强包括：mixup, 随机颜色失真，随机剪裁，随机扩张，随机插值法，随机翻转。Yolo v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型测试不同尺寸图片，我们分别提供了尺寸为608/416/320大小的测试结果。

### RetinaNet

|   骨架网络        | 每张GPU图片个数 | 学习率策略 | Box AP | 下载  |
| :---------------: | :-----: | :-----: | :----: | :-------: |
| ResNet50-FPN      |    2    |   1x    |  36.0  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar)  |
| ResNet101-FPN     |    2    |   1x    |  37.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) |
| ResNeXt101-vd-FPN |    1    |   1x    |  40.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_x101_vd_64x4d_fpn_1x.tar) |

**注意事项:** RetinaNet系列模型中，在总batch size为16下情况下，初始学习率改为0.01。

### SSD

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略|推理时间(fps) | Box AP | 下载 |
| :----------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: |
| VGG16        | 300  |     8   |   40万  |    81.613     |  25.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar) |
| VGG16        | 512  |     8   |   40万  |    46.007     |  29.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512.tar) |

**注意事项:** VGG-SSD在总batch size为32下训练40万轮。

### SSD 基于Pascal VOC数据集

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载  |
| :----------- | :--: | :-----: | :-----: |  :------------: |:----: | :-------: |
| MobileNet v1 | 300  |    32   |   120e  |     159.543     | 73.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) |
| VGG16        | 300  |     8   |   240e  |     117.279     | 77.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300_voc.tar) |
| VGG16        | 512  |     8   |   240e  |      65.975     | 80.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512_voc.tar) |

**注意事项:** MobileNet-SSD在2卡，总batch size为64下训练120周期。VGG-SSD在总batch size为32下训练240周期。数据增强包括：随机颜色失真，随机剪裁，随机扩张，随机翻转。

## 人脸检测

详细请参考[人脸检测模型](../configs/face_detection).
