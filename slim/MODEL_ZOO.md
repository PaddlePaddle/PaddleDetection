# 压缩模型库

## 测试环境

- Python 2.7.1
- PaddlePaddle >=1.6
- CUDA 9.0
- cuDNN >=7.4
- NCCL 2.1.2

## 裁剪模型库

### 训练策略

- 裁剪模型训练时使用[PaddleDetection模型库](../../docs/MODEL_ZOO_cn.md)发布的模型权重作为预训练权重。
- 裁剪训练使用模型默认配置，即除`pretrained_weights`外配置不变。
- 裁剪模型全部为基于敏感度的卷积通道裁剪。
- YOLOv3模型主要裁剪`yolo_head`部分，即裁剪参数如下。

```
--pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights"
```
- YOLOv3模型裁剪中裁剪策略`r578`表示`yolo_head`中三个输出分支一次使用`0.5, 0.7, 0.8`的裁剪率裁剪，即裁剪率如下。

```
--pruned_ratios="0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.8"
```

- YOLOv3模型裁剪中裁剪策略`sensity`表示`yolo_head`中各参数裁剪率如下，该裁剪率为使用`yolov3_mobilnet_v1`模型在COCO数据集上敏感度实验分析得出。

```
--pruned_ratios="0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.3,0.3,0.3,0.2,0.1,0.3,0.4,0.4,0.4,0.4,0.3"
```

### YOLOv3 on COCO

| 骨架网络         |  裁剪策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-------: | :------: |:------: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |  sensity  |   320    |  39.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50_dcn_prune1x.tar) |
| MobileNetV1      |   r578    |   608    |  27.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |   416    |  26.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   r578    |   320    |  24.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |

### YOLOv3 on Pascal VOC

| 骨架网络         |  裁剪策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :-------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      |   r578    |   608    |  77.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |   416    |  77.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   r578    |   320    |  75.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |


## 蒸馏模型库

### 训练策略

- 蒸馏模型训练时teacher模型使用[PaddleDetection模型库](../../docs/MODEL_ZOO_cn.md)发布的模型权重作为预训练权重。
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
