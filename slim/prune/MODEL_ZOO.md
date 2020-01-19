# 裁剪模型库

## 测试环境

- Python 2.7.1
- PaddlePaddle >=1.6
- CUDA 9.0
- cuDNN >=7.4
- NCCL 2.1.2

## 通用设置

- 裁剪模型训练时使用[PaddleDetection模型库](../../docs/MODEL_ZOO_cn.md)发布的模型权重作为预训练权重。
- 裁剪训练使用模型默认配置，即除`pretrained_weights`外配置不变。
- 裁剪模型全部为基于敏感度的卷积通道裁剪。

## 训练策略

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


## 基线

### YOLOv3 on COCO

| 骨架网络         |  数据集  |  裁剪策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :------: | :-------: | :------: |:------: | :-----------------------------------------------------: |
| ResNet50-vd-dcn  |   COCO   |  sensity  |   320    |  39.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_r50_dcn_prune1x.tar) |
| MobileNetV1      |   COCO   |   r578    |   608    |  27.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   COCO   |   r578    |   416    |  26.8   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |
| MobileNetV1      |   COCO   |   r578    |   320    |  24.0   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_prune578.tar) |

### YOLOv3 on Pascal VOC

| 骨架网络         |  数据集  |  裁剪策略 | 输入尺寸 | Box AP  |                           下载                          |
| :----------------| :------: | :-------: | :------: |:------: | :-----------------------------------------------------: |
| MobileNetV1      |   VOC    |   r578    |   608    |  77.6   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   VOC    |   r578    |   416    |  77.7   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
| MobileNetV1      |   VOC    |   r578    |   320    |  75.5   | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/prune/yolov3_mobilenet_v1_voc_prune578.tar) |
