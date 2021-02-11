# 压缩benchmark

在PaddleDetection, 提供了基于PaddleSlim进行模型压缩的完整教程和实验结果。详细教程请参考：

- [量化](quantization)
- [裁剪](prune)
- [蒸馏](distillation)
- [搜索](nas)

下面给出压缩的benchmark实验结果。

## 测试环境

- Python 2.7.1
- PaddlePaddle >=1.6
- CUDA 9.0
- cuDNN >=7.4
- NCCL 2.1.2

## 剪裁模型库

### 训练策略

- 剪裁模型训练时使用[PaddleDetection模型库](https://paddledetection.readthedocs.io/MODEL_ZOO_cn.html)发布的模型权重作为预训练权重。
- 剪裁训练使用模型默认配置，即除`pretrained_weights`外配置不变。
- 剪裁模型全部为基于敏感度的卷积通道剪裁。
- YOLOv3模型主要剪裁`yolo_head`部分，即剪裁参数如下。

```
--pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights"
```
- YOLOv3模型剪裁中剪裁策略`r578`表示`yolo_head`中三个输出分支一次使用`0.5, 0.7, 0.8`的剪裁率剪裁，即剪裁率如下。

```
--pruned_ratios="0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.8"
```

- YOLOv3模型剪裁中剪裁策略`sensity`表示`yolo_head`中各参数剪裁率如下，该剪裁率为使用`yolov3_mobilnet_v1`模型在COCO数据集上敏感度实验分析得出。

```
--pruned_ratios="0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.3,0.3,0.3,0.2,0.1,0.3,0.4,0.4,0.4,0.4,0.3"
```

### YOLOv3 on COCO

| 骨架网络         |  剪裁策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 |   Box AP   |                           下载                          |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |  baseline | 44.71          | 176.82          |   608    | 39.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) |
| ResNet50-vd-dcn  |  sensity  | 37.53(-16.06%) | 149.49(-15.46%) |   608    | 39.8(+0.7) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune1x.tar) |
| ResNet50-vd-dcn  |   r578    | 29.98(-32.94%) | 112.08(-36.61%) |   608    | 38.3(-0.8) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune578.tar) |
| MobileNetV1      |  baseline | 20.64          |  94.60          |   608    | 29.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  baseline |  9.66          |  94.60          |   416    | 29.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  baseline |  5.72          |  94.60          |   320    | 27.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  sensity  | 13.57(-34.27%) |  67.60(-28.54%) |   608    | 30.2(+0.9) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |  sensity  |  6.35(-34.27%) |  67.60(-28.54%) |   416    | 29.7(+0.4) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |  sensity  |  3.76(-34.27%) |  67.60(-28.54%) |   320    | 27.2(+0.1) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |   r578    |  6.27(-69.64%) |  31.30(-66.90%) |   608    | 27.8(-1.5) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |  2.93(-69.64%) |  31.30(-66.90%) |   416    | 26.8(-2.5) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |  1.74(-69.64%) |  31.30(-66.90%) |   320    | 24.0(-3.1) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`sensity`剪裁策略下，`YOLOv3-ResNet50-vd-dcn`和`YOLOv3-MobileNetV1`分别减少了`16.06%`和`34.27%`的FLOPs，输入图像尺寸为608时精度分别提高`0.7`和`0.9`
- 在使用`r578`剪裁策略下，`YOLOv3-ResNet50-vd-dcn`和`YOLOv3-MobileNetV1`分别减少了`32.98%`和`69.64%`的FLOPs，输入图像尺寸为608时精度分别降低`0.8`和`1.5`

### YOLOv3 on Pascal VOC

| 骨架网络         |  剪裁策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 |   Box AP   |                           下载                          |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------: | :-----------------------------------------------------: |
| MobileNetV1      |  baseline | 20.20          |  93.37          |   608    | 76.2       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  baseline |  9.46          |  93.37          |   416    | 76.7       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  baseline |  5.60          |  93.37          |   320    | 75.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  sensity  | 13.22(-34.55%) |  66.53(-28.74%) |   608    | 78.4(+2.2) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |  sensity  |  6.19(-34.55%) |  66.53(-28.74%) |   416    | 78.7(+2.0) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |  sensity  |  3.66(-34.55%) |  66.53(-28.74%) |   320    | 76.1(+0.8) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |   r578    |  6.15(-69.57%) |  30.81(-67.00%) |   608    | 77.6(+1.4) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |  2.88(-69.57%) |  30.81(-67.00%) |   416    | 77.7(+1.0) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |  1.70(-69.57%) |  30.81(-67.00%) |   320    | 75.5(+0.2) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`sensity`和`r578`剪裁策略下，`YOLOv3-MobileNetV1`分别减少了`34.55%`和`69.57%`的FLOPs，输入图像尺寸为608时精度分别提高`2.2`和`1.4`

### 蒸馏通道剪裁模型

可通过高精度模型蒸馏通道剪裁后模型的方式，训练方法及相关示例见[蒸馏通道剪裁模型](https://github.com/PaddlePaddle/PaddleDetection/blob/master/slim/extensions/distill_pruned_model/distill_pruned_model_demo.ipynb)。

COCO数据集上蒸馏通道剪裁模型库如下。

| 骨架网络         |  剪裁策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 |         teacher模型          |   Box AP   |                           下载                          |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------------------------: | :--------: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |  baseline | 44.71          | 176.82          |   608    |              -               | 39.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) |
| ResNet50-vd-dcn  |   r578    | 29.98(-32.94%) | 112.08(-36.61%) |   608    | YOLOv3-ResNet50-vd-dcn(39.1) | 39.7(+0.6) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune578_distill.tar) |
| MobileNetV1      |  baseline | 20.64          |  94.60          |   608    |              -               | 29.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  baseline |  9.66          |  94.60          |   416    |              -               | 29.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  baseline |  5.72          |  94.60          |   320    |              -               | 27.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |   r578    |  6.27(-69.64%) |  31.30(-66.90%) |   608    | YOLOv3-ResNet34(36.2)        | 29.0(-0.3) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |  2.93(-69.64%) |  31.30(-66.90%) |   416    | YOLOv3-ResNet34(34.3)        | 28.0(-1.3) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |  1.74(-69.64%) |  31.30(-66.90%) |   320    | YOLOv3-ResNet34(31.4)        | 25.1(-2.0) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`r578`剪裁策略并使用`YOLOv3-ResNet50-vd-dcn`作为teacher模型蒸馏，`YOLOv3-ResNet50-vd-dcn`模型减少了`32.94%`的FLOPs，输入图像尺寸为608时精度提高`0.6`
- 在使用`r578`剪裁策略并使用`YOLOv3-ResNet34`作为teacher模型蒸馏下，`YOLOv3-MobileNetV1`模型减少了`69.64%`的FLOPs，输入图像尺寸为608时精度降低`0.3`

Pascal VOC数据集上蒸馏通道剪裁模型库如下。

| 骨架网络         |  剪裁策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 |       teacher模型      |   Box AP   |                           下载                          |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------------------: | :--------: | :-----------------------------------------------------: |
| MobileNetV1      |  baseline | 20.20          |  93.37          |   608    |           -            | 76.2       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  baseline |  9.46          |  93.37          |   416    |           -            | 76.7       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  baseline |  5.60          |  93.37          |   320    |           -            | 75.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |   r578    |  6.15(-69.57%) |  30.81(-67.00%) |   608    | YOLOv3-ResNet34(82.6)  | 78.8(+2.6) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |  2.88(-69.57%) |  30.81(-67.00%) |   416    | YOLOv3-ResNet34(81.9)  | 78.7(+2.0) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |  1.70(-69.57%) |  30.81(-67.00%) |   320    | YOLOv3-ResNet34(80.1)  | 76.3(+2.0) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`r578`剪裁策略并使用`YOLOv3-ResNet34`作为teacher模型蒸馏下，`YOLOv3-MobileNetV1`模型减少了`69.57%`的FLOPs，输入图像尺寸为608时精度提高`2.6`

### YOLOv3通道剪裁模型推理时延

- 时延单位均为`ms/images`
- Tesla P4时延为单卡并开启TensorRT推理时延
- 高通835/高通855/麒麟970时延为使用PaddleLite部署，使用`arm8`架构并使用4线程(4 Threads)推理时延

| 骨架网络         | 数据集 | 剪裁策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 |    Tesla P4     |     麒麟970      |     高通835      |     高通855      |
| :--------------- | :----: | :------: | :------------: | :-------------: | :------: | :-------------: | :--------------: | :--------------: | :--------------: |
| MobileNetV1      |  VOC   | baseline | 20.20          |  93.37          |   608    | 16.556          | 748.404          | 734.970          | 289.878          |
| MobileNetV1      |  VOC   | baseline |  9.46          |  93.37          |   416    |  9.031          | 371.214          | 349.065          | 140.877          |
| MobileNetV1      |  VOC   | baseline |  5.60          |  93.37          |   320    |  6.235          | 221.705          | 200.498          |  80.515          |
| MobileNetV1      |  VOC   |   r578   |  6.15(-69.57%) |  30.81(-67.00%) |   608    | 10.064(-39.21%) | 314.531(-57.97%) | 323.537(-55.98%) | 123.414(-57.43%) |
| MobileNetV1      |  VOC   |   r578   |  2.88(-69.57%) |  30.81(-67.00%) |   416    |  5.478(-39.34%) | 151.562(-59.17%) | 146.014(-58.17%) |  56.420(-59.95%) |
| MobileNetV1      |  VOC   |   r578   |  1.70(-69.57%) |  30.81(-67.00%) |   320    |  3.880(-37.77%) |  91.132(-58.90%) |  87.440(-56.39%) |  31.470(-60.91%) |
| ResNet50-vd-dcn  |  COCO  | baseline | 44.71          | 176.82          |   608    | 36.127          |        -         |        -         |        -         |
| ResNet50-vd-dcn  |  COCO  | sensity  | 37.53(-16.06%) | 149.49(-15.46%) |   608    | 33.245(-7.98%)  |        -         |        -         |        -         |
| ResNet50-vd-dcn  |  COCO  |   r578   | 29.98(-32.94%) | 112.08(-36.61%) |   608    | 29.138(-19.35%) |        -         |        -         |        -         |

- 在使用`r578`剪裁策略下，`YOLOv3-MobileNetV1`模型减少了`69.57%`的FLOPs，输入图像尺寸为608时在单卡Tesla P4(TensorRT)推理时间减少`39.21%`，在麒麟970/高通835/高通855上推理时延分别减少`57.97%`, `55.98%`和`57.43%`
- 在使用`sensity`和`r578`剪裁策略下，`YOLOv3-ResNet50-vd-dcn`模型分别减少了`16.06%`和`32.94%`的FLOPs，输入图像尺寸为608时在单卡Tesla P4(TensorRT)推理时间分别减少`7.98%`和`19.35%`

## 蒸馏模型库

### 训练策略

- 蒸馏模型训练时teacher模型使用[PaddleDetection模型库](https://paddledetection.readthedocs.io/zh/latest/MODEL_ZOO_cn.html)发布的模型权重作为预训练权重。
- 蒸馏模型训练时student模型使用backbone的预训练权重
- 蒸馏策略`l2_distiil`为使用teacher模型和student模型特征图的L2损失作为蒸馏损失进行蒸馏，为`slim/distillation/distill.py`的默认策略
- 蒸馏策略`split_distiil`为使用YOLOv3细粒度损失进行蒸馏，通过`-o use_fine_grained_loss=true`指定

### YOLOv3 on COCO

| 骨架网络         |    蒸馏策略   | 输入尺寸 |       teacher模型      |    Box AP    |                           下载                          |
| :----------------| :-----------: | :------: | :--------------------: | :----------: | :-----------------------------------------------------: |
| MobileNetV1      |    baseline   |   608    |           -            | 29.3         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |    baseline   |   416    |           -            | 29.3         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |    baseline   |   320    |           -            | 27.1         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      | split_distiil |   608    | YOLOv3-ResNet34(36.2)  | 31.4(+2.1)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |
| MobileNetV1      | split_distiil |   416    | YOLOv3-ResNet34(34.3)  | 30.0(+0.7)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |
| MobileNetV1      | split_distiil |   320    | YOLOv3-ResNet34(31.4)  | 27.1(+0.0)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`YOLOv3-ResNet34`模型通过`split_distiil`策略蒸馏下，输入图像尺寸为608时`YOLOv3-MobileNetV1`模型精度提高`2.1`

### YOLOv3 on Pascal VOC

| 骨架网络         |    蒸馏策略   | 输入尺寸 |       teacher模型      |   Box AP   |                           下载                          |
| :----------------| :-----------: | :------: | :--------------------: | :--------: | :-----------------------------------------------------: |
| MobileNetV1      |    baseline   |   608    |           -            | 76.2       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |    baseline   |   416    |           -            | 76.7       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |    baseline   |   320    |           -            | 75.3       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNetV1      |  l2_distiil   |   608    | YOLOv3-ResNet34(82.6)  | 79.0(+2.8) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNetV1      |  l2_distiil   |   416    | YOLOv3-ResNet34(81.9)  | 78.2(+1.5) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNetV1      |  l2_distiil   |   320    | YOLOv3-ResNet34(80.1)  | 75.5(+0.2) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中`YOLOv3-MobileNetV1`提供了在`608/416/320`三种不同尺寸下的精度结果
- 在使用`YOLOv3-ResNet34`模型通过`l2_distiil`策略蒸馏下，输入图像尺寸为608时`YOLOv3-MobileNetV1`模型精度提高`2.8`

## 量化模型库

### 训练策略

- 量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。

### YOLOv3 on COCO

| 骨架网络         | 预训练权重 | 量化策略 | 输入尺寸 |   Box AP   |                           下载                          |
| :----------------| :--------: | :------: | :------: | :--------: | :-----------------------------------------------------: |
| MobileNetV1      |  ImageNet  | baseline |   608    | 29.3         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  ImageNet  | baseline |   416    | 29.3         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  ImageNet  | baseline |   320    | 27.1         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNetV1      |  ImageNet  |   post   |   608    | 27.9(-1.4)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   416    | 28.0(-1.3)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   320    | 26.0(-1.1)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   608    | 28.1(-1.2)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   416    | 28.2(-1.1)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   320    | 25.8(-1.3)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  | baseline |   608    | 36.2         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34         |  ImageNet  | baseline |   416    | 34.3         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34         |  ImageNet  | baseline |   320    | 31.4         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34         |  ImageNet  |   post   |   608    | 35.7(-0.5)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_post.tar) |
| ResNet34         |  ImageNet  |  aware   |   608    | 35.2(-1.1)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   416    | 33.3(-1.0)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   320    | 30.3(-1.1)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| R50vd-dcn        | object365  | baseline |   608    | 41.4         | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |
| R50vd-dcn        | object365  |  aware   |   608    | 40.6(-0.8)   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   416    | 37.5(-)      | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   320    | 34.1(-)      | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |

- YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型权重不同尺寸图片，表中部分模型提供了在`608/416/320`三种不同尺寸下的精度结果
- `YOLOv3-MobileNetV1`使用离线(post)和在线(aware)两种量化方式，输入图像尺寸为608时精度分别降低`1.4`和`1.2`
- `YOLOv3-ResNet34`使用离线(post)和在线(aware)两种量化方式，输入图像尺寸为608时精度分别降低`0.5`和`1.1`
- `YOLOv3-R50vd-dcn`使用在线(aware)量化方式，输入图像尺寸为608时精度降低`0.8`

### BlazeFace on WIDER FACE

| 模型             | 量化策略 | 输入尺寸 |  Easy Set  | Medium Set |  Hard Set  |                           下载                          |
| :--------------- | :------: | :------: | :--------: | :--------: | :--------: | :-----------------------------------------------------: |
| BlazeFace        | baseline |   640    | 91.5       | 89.2       | 79.7       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_original.tar) |
| BlazeFace        |   post   |   640    | 87.8(-3.7) | 85.1(-3.9) | 74.9(-4.8) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_post.tar) |
| BlazeFace        |  aware   |   640    | 90.5(-1.0) | 87.9(-1.3) | 77.6(-2.1) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_aware.tar) |
| BlazeFace-Lite   | baseline |   640    | 90.9       | 88.5       | 78.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_lite.tar) |
| BlazeFace-Lite   |   post   |   640    | 89.4(-1.5) | 86.7(-1.8) | 75.7(-2.4) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_post.tar) |
| BlazeFace-Lite   |  aware   |   640    | 89.7(-1.2) | 87.3(-1.2) | 77.0(-1.1) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_aware.tar) |
| BlazeFace-NAS    | baseline |   640    | 83.7       | 80.7       | 65.8       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_nas.tar) |
| BlazeFace-NAS    |   post   |   640    | 81.6(-2.1) | 78.3(-2.4) | 63.6(-2.2) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_post.tar) |
| BlazeFace-NAS    |  aware   |   640    | 83.1(-0.6) | 79.7(-1.0) | 64.2(-1.6) | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_aware.tar) |

- `BlazeFace`系列模型中在线(aware)量化性能明显优于离线(post)量化
- `BlazeFace`模型使用在线(aware)量化方式，在`Easy/Medium/Hard`数据集上精度分别降低`1.0`, `1.3`和`2.1`
- `BlazeFace-Lite`模型使用在线(aware)量化方式，在`Easy/Medium/Hard`数据集上精度分别降低`1.2`, `1.2`和`1.1`
- `BlazeFace-NAS`模型使用在线(aware)量化方式，在`Easy/Medium/Hard`数据集上精度分别降低`0.6`, `1.0`和`1.6`
