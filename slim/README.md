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

- 剪裁模型训练时使用[PaddleDetection模型库](https://paddledetection.readthedocs.io/zh/latest/MODEL_ZOO_cn.html)发布的模型权重作为预训练权重。
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

| 骨架网络         |  剪裁策略 | FLOPs剪裁率 | 模型体积剪裁率 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-------: | :---------: | :------------: | :------: | :-----: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |  sensity  |   18.41%    |     15.46%     |   608    |  39.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune1x.tar) |
| ResNet50-vd-dcn  |   r578    |   43.69%    |     36.61%     |   608    |  38.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune578.tar) |
| MobileNetV1      |  sensity  |   28.76%    |     28.54%     |   608    |  30.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |  sensity  |   28.76%    |     28.54%     |   416    |  29.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |  sensity  |   28.76%    |     28.54%     |   320    |  27.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune1x.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   608    |  27.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   416    |  26.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   320    |  24.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |

### YOLOv3 on Pascal VOC

| 骨架网络         |  剪裁策略 | FLOPs剪裁率 | 模型体积剪裁率 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-------: | :---------: | :------------: | :------: | :-----: | :-----------------------------------------------------: |
| MobileNetV1      |  sensity  |   34.55%    |     28.75%     |   608    |  78.4   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |  sensity  |   34.55%    |     28.75%     |   416    |  78.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |  sensity  |   34.55%    |     28.75%     |   320    |  76.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune1x.tar) |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   608    |  77.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   416    |  77.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   320    |  75.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |

### 蒸馏通道剪裁模型

可通过高精度模型蒸馏通道剪裁后模型的方式，训练方法及相关示例见[蒸馏通道剪裁模型](https://github.com/PaddlePaddle/PaddleDetection/blob/master/slim/extensions/distill_pruned_model/distill_pruned_model_demo.ipynb)。

COCO数据集上蒸馏通道剪裁模型库如下。

| 骨架网络         |  剪裁策略 | FLOPs剪裁率 | 模型体积剪裁率 | 输入尺寸 |      teacher模型       | Box AP  |                           下载                          |
| :----------------| :-------: | :---------: | :------------: | :------: | :--------------------- | :-----: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |   r578    |   43.69%    |     36.61%     |   608    | YOLOv3-ResNet50-vd-dcn |  39.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50vd_dcn_prune578_distill.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   608    | YOLOv3-ResNet34        |  29.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   416    | YOLOv3-ResNet34        |  28.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |   67.56%    |     66.90%     |   320    | YOLOv3-ResNet34        |  25.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578_distillby_r34.tar) |

Pascal VOC数据集上蒸馏通道剪裁模型库如下。

| 骨架网络         |  剪裁策略 | FLOPs剪裁率 | 模型体积剪裁率 | 输入尺寸 |      teacher模型       | Box AP  |                           下载                          |
| :----------------| :-------: | :---------: | :------------: | :------: | :--------------------- | :-----: | :-----------------------------------------------------: |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   608    | YOLOv3-ResNet34        |  78.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   416    | YOLOv3-ResNet34        |  78.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |
| MobileNetV1      |   r578    |   69.57%    |     67.00%     |   320    | YOLOv3-ResNet34        |  76.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578_distillby_r34.tar) |

### YOLOv3通道剪裁模型推理时延

- 时延单位均为`ms/images`
- Tesla P4时延为单卡并开启TensorRT推理时延
- 高通835/高通855/麒麟970时延为使用PaddleLite部署，使用`arm8`架构并使用4线程(4 Threads)推理时延

| 骨架网络         | 数据集 | 剪裁策略 | FLOPs剪裁率 | 模型体积剪裁率 | 输入尺寸 | Tesla P4 | 麒麟970 | 高通835 | 高通855 |
| :--------------- | :----: | :------: | :---------: | :------------: | :------: | :------: | :-----: | :-----: | :-----: |
| MobileNetV1      |  VOC   | baseline |      -      |       -        |   608    |  16.556  | 748.404 | 734.970 | 289.878 |
| MobileNetV1      |  VOC   | baseline |      -      |       -        |   416    |   9.031  | 371.214 | 349.065 | 140.877 |
| MobileNetV1      |  VOC   | baseline |      -      |       -        |   320    |   6.235  | 221.705 | 200.498 |  80.515 |
| MobileNetV1      |  VOC   |   r578   |   69.57%    |     67.00%     |   608    |  10.064  | 314.531 | 323.537 | 123.414 |
| MobileNetV1      |  VOC   |   r578   |   69.57%    |     67.00%     |   416    |   5.478  | 151.562 | 146.014 |  56.420 |
| MobileNetV1      |  VOC   |   r578   |   69.57%    |     67.00%     |   320    |   3.880  |  91.132 |  87.440 |  31.470 |
| ResNet50-vd-dcn  |  COCO  | baseline |      -      |       -        |   608    |  36.127  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  | baseline |      -      |       -        |   416    |  20.437  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  | baseline |      -      |       -        |   320    |  14.037  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  | sensity  |   18.41%    |     15.46%     |   608    |  33.245  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  | sensity  |   18.41%    |     15.46%     |   416    |  19.246  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  | sensity  |   18.41%    |     15.46%     |   320    |  13.656  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  |   r578   |   43.69%    |     36.61%     |   608    |  29.138  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  |   r578   |   43.69%    |     36.61%     |   416    |  16.439  |    -    |    -    |    -    |
| ResNet50-vd-dcn  |  COCO  |   r578   |   43.69%    |     36.61%     |   320    |  11.339  |    -    |    -    |    -    |


## 蒸馏模型库

### 训练策略

- 蒸馏模型训练时teacher模型使用[PaddleDetection模型库](https://paddledetection.readthedocs.io/zh/latest/MODEL_ZOO_cn.html)发布的模型权重作为预训练权重。
- 蒸馏模型训练时student模型使用backbone的预训练权重

### YOLOv3 on COCO

| 骨架网络         |    蒸馏策略   | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-----------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      | split_distiil |   608    |  31.4   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |
| MobileNetV1      | split_distiil |   416    |  30.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |
| MobileNetV1      | split_distiil |   320    |  27.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |

### YOLOv3 on Pascal VOC

| 骨架网络         |    蒸馏策略   | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-----------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      |  l2_distiil   |   608    |  79.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNetV1      |  l2_distiil   |   416    |  78.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNetV1      |  l2_distiil   |   320    |  75.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |

## 量化模型库

### 训练策略

- 量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。

### YOLOv3 on COCO

| 骨架网络         | 预训练权重 | 量化策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :--------: | :------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      |  ImageNet  |   post   |   608    |  27.9   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   416    |  28.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |   post   |   320    |  26.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   608    |  28.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   416    |  28.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| MobileNetV1      |  ImageNet  |  aware   |   320    |  25.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |   post   |   608    |  35.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_post.tar) |
| ResNet34         |  ImageNet  |  aware   |   608    |  35.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   416    |  33.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| ResNet34         |  ImageNet  |  aware   |   320    |  30.3   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   608    |  40.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   416    |  37.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |
| R50vd-dcn        | object365  |  aware   |   320    |  34.1   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |

### BlazeFace on WIDER FACE

| 模型             | 量化策略 | 输入尺寸 | Easy Set | Medium Set | Hard Set |                           下载                          |
| :--------------- | :------: | :------: | :------: | :--------: | :------: | :-----------------------------------------------------: |
| BlazeFace        |   post   |   640    |   87.8   |    85.1    |   74.9   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_post.tar) |
| BlazeFace        |  aware   |   640    |   90.5   |    87.9    |   77.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_aware.tar) |
| BlazeFace-Lite   |   post   |   640    |   89.4   |    86.7    |   75.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_post.tar) |
| BlazeFace-Lite   |  aware   |   640    |   89.7   |    87.3    |   77.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_aware.tar) |
| BlazeFace-NAS    |   post   |   640    |   81.6   |    78.3    |   63.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_post.tar) |
| BlazeFace-NAS    |  aware   |   640    |   83.1   |    79.7    |   64.2   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_aware.tar) |
