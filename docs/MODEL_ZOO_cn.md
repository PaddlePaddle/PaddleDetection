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

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     12.747     |  35.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_1x.yml) |
| ResNet50             | Faster         |    1    |   2x    |     12.686     |  37.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_2x.yml) |
| ResNet50             | Mask           |    1    |   1x    |     11.615     |  36.5  |  32.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r50_1x.yml) |
| ResNet50             | Mask           |    1    |   2x    |     11.494     |  38.2  |  33.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r50_2x.yml) |
| ResNet50-vd          | Faster         |    1    |   1x    |     12.575     |  36.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_vd_1x.yml) |
| ResNet34-FPN            | Faster         |     2     |   1x    |     -     |  36.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r34_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r34_fpn_1x.yml) |
| ResNet34-vd-FPN            | Faster         |     2     |   1x    |     -     |  37.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r34_vd_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r34_vd_fpn_1x.yml) |
| ResNet50-FPN         | Faster         |    2    |   1x    |     22.273     |  37.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_fpn_1x.yml) |
| ResNet50-FPN         | Faster         |    2    |   2x    |     22.297     |  37.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_fpn_2x.yml) |
| ResNet50-FPN         | Mask           |    1    |   1x    |     15.184     |  37.9  |  34.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r50_fpn_1x.yml) |
| ResNet50-FPN         | Mask           |    1    |   2x    |     15.881     |  38.7  |  34.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r50_fpn_2x.yml) |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |     17.507     |  40.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/cascade_rcnn_r50_fpn_1x.yml) |
| ResNet50-FPN         | Cascade Mask   |    1    |   1x    |       12.43        |  41.3  |  35.5   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/cascade_mask_rcnn_r50_fpn_1x.yml) |
| ResNet50-vd-FPN      | Faster         |    2    |   2x    |     21.847     |  38.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r50_vd_fpn_2x.yml) |
| ResNet50-vd-FPN      | Mask           |    1    |   2x    |     15.825     |  39.8  |  35.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r50_vd_fpn_2x.yml) |
| CBResNet50-vd-FPN         | Faster         |     2     |   1x    |     -     |  39.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_cbr50_vd_dual_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_cbr50_vd_dual_fpn_1x.yml) |
| ResNet101            | Faster         |    1    |   1x    |     9.316      |  38.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r101_1x.yml) |
| ResNet101-FPN        | Faster         |    1    |   1x    |     17.297     |  38.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r101_fpn_1x.yml) |
| ResNet101-FPN        | Faster         |    1    |   2x    |     17.246     |  39.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r101_fpn_2x.yml) |
| ResNet101-FPN        | Mask           |    1    |   1x    |     12.983     |  39.5  |  35.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r101_fpn_1x.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |     17.011     |  40.5  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r101_vd_fpn_1x.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |     16.934     |  40.8  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_r101_vd_fpn_2x.yml) |
| ResNet101-vd-FPN     | Mask           |    1    |   1x    |     13.105     |  41.4  |  36.8   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_r101_vd_fpn_1x.yml) |
| CBResNet101-vd-FPN         | Faster         |     2     |   1x    |     -     |  42.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_cbr101_vd_dual_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_cbr101_vd_dual_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |     8.815      |  42.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_x101_vd_64x4d_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |     8.809      |  41.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_x101_vd_64x4d_fpn_2x.yml) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   1x    |     7.689      |  42.9  |  37.9   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_x101_vd_64x4d_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   2x    |     7.859      |  42.6  |  37.6   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_x101_vd_64x4d_fpn_2x.yml) |
| SENet154-vd-FPN      | Faster         |    1    |  1.44x  |     3.408      |  42.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_vd_fpn_s1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/faster_rcnn_se154_vd_fpn_s1x.yml) |
| SENet154-vd-FPN      | Mask           |    1    |  1.44x  |     3.233      |  44.0  |  38.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/mask_rcnn_se154_vd_fpn_s1x.yml) |
| ResNet101-vd-FPN            | CascadeClsAware Faster   |     2     |   1x    |     -     |  44.7(softnms)  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r101_vd_fpn_1x_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/cascade_rcnn_cls_aware_r101_vd_fpn_1x_softnms.yml) |
| ResNet101-vd-FPN            | CascadeClsAware Faster   |     2     |   1x    |     -     |  46.5(multi-scale test)  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r101_vd_fpn_1x_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/cascade_rcnn_cls_aware_r101_vd_fpn_1x_softnms.yml) |

### Deformable 卷积网络v2

| 骨架网络             | 网络类型           | 卷积    | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | Mask AP |                           下载                           | 配置文件 |
| :------------------- | :------------- | :-----: |:--------: | :-----: | :-----------: |:----: | :-----: | :----------------------------------------------------------: | :----: |
| ResNet50-FPN         | Faster         | c3-c5   |    2      |   1x    |    19.978     |  41.0  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/faster_rcnn_dcn_r50_fpn_1x.yml) |
| ResNet50-vd-FPN      | Faster         | c3-c5   |    2      |   2x    |    19.222     |  42.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/faster_rcnn_dcn_r50_vd_fpn_2x.yml) |
| ResNet101-vd-FPN     | Faster         | c3-c5   |    2      |   1x    |    14.477     |  44.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r101_vd_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/faster_rcnn_dcn_r101_vd_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Faster         | c3-c5   |    1      |   1x    |    7.209      |  45.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.yml) |
| ResNet50-FPN         | Mask           | c3-c5   |    1      |   1x    |    14.53      |  41.9  |  37.3   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/mask_rcnn_dcn_r50_fpn_1x.yml) |
| ResNet50-vd-FPN      | Mask           | c3-c5   |    1      |   2x    |    14.832     |  42.9  |  38.0   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_vd_fpn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/mask_rcnn_dcn_r50_vd_fpn_2x.yml) |
| ResNet101-vd-FPN     | Mask           | c3-c5   |    1      |   1x    |    11.546     |  44.6  |  39.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r101_vd_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/mask_rcnn_dcn_r101_vd_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Mask           | c3-c5   |    1      |   1x    |     6.45      |  46.2  |  40.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.yml) |
| ResNet50-FPN         | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  44.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_dcn_r50_fpn_1x.yml) |
| ResNet101-vd-FPN     | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  46.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_dcn_r101_vd_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Cascade Faster | c3-c5   |    2      |   1x    |      -        |  47.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x.yml) |
| SENet154-vd-FPN      | Cascade Mask   | c3-c5   |    1      |  1.44x  |      -        |  51.9  |  43.9   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_dcnv2_se154_vd_fpn_gn_s1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_mask_rcnn_dcnv2_se154_vd_fpn_gn_s1x.yml) |
| ResNet200-vd-FPN-Nonlocal   | CascadeClsAware Faster   | c3-c5 |     1     |   2.5x    |     3.103     |  51.7%(softnms)  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.yml) |
| CBResNet200-vd-FPN-Nonlocal   | Cascade Faster  | c3-c5 |     1     |   2.5x    |     1.68     |  53.3%(softnms)  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.yml) |

**注意事项:**  

- Deformable卷积网络v2(dcn_v2)参考自论文[Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5`意思是在resnet模块的3到5阶段增加`dcn`.
- 详细的配置文件在[configs/dcn](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/dcn)


### HRNet
* 详情见[HRNet模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/hrnet/)。


### Res2Net
* 详情见[Res2Net模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/res2net/)。

### IOU loss
* 目前模型库中包括GIOU loss和DIOU loss，详情见[IOU loss模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master//configs/iou_loss/).

### GCNet
* 详情见[GCNet模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/gcnet/).

### Libra R-CNN
* 详情见[Libra R-CNN模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/libra_rcnn/).

### Auto Augmentation
* 详情见[Auto Augmentation模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/autoaugment/).

### Group Normalization

| 骨架网络             | 网络类型           | 每张GPU图片个数 | 学习率策略 | Box AP | Mask AP |                           下载                           | 配置文件 |
| :------------------- | :------------- |:--------: | :-----: | :----: | :-----: | :----------------------------------------------------------: | :----: |
| ResNet50-FPN         | Faster         |    2      |   2x    |  39.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_gn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/gn/faster_rcnn_r50_fpn_gn_2x.yml) |
| ResNet50-FPN         | Mask           |    1      |   2x    |  40.1  |   35.8  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_gn_2x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn_2x.yml) |

**注意事项:**

- Group Normalization参考论文[Group Normalization](https://arxiv.org/abs/1803.08494).
- 详细的配置文件在[configs/gn](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/gn)

### YOLO v3

| 骨架网络     | 预训练数据集 | 输入尺寸 | 加入deformable卷积 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 | 配置文件 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: | :-------: | :-------: |
| DarkNet53 (paper)   | ImageNet | 608  |  否    |    8    |   270e  |      -        |  33.0  | - | - |
| DarkNet53 (paper)   | ImageNet | 416  |  否    |    8    |   270e  |      -        |  31.0  | - | - |
| DarkNet53 (paper)   | ImageNet | 320  |  否    |    8    |   270e  |      -        |  28.2  | - | - |
| DarkNet53           | ImageNet | 608  |  否    |    8    |   270e  |    45.571     |  38.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet.yml) |
| DarkNet53           | ImageNet | 416  |  否    |    8    |   270e  |      -        |  37.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet.yml) |
| DarkNet53           | ImageNet | 320  |  否    |    8    |   270e  |      -        |  34.8  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet.yml) |
| MobileNet-V1        | ImageNet | 608  |  否    |    8    |   270e  |    78.302     |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1.yml) |
| MobileNet-V1        | ImageNet | 416  |  否    |    8    |   270e  |      -        |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1.yml) |
| MobileNet-V1        | ImageNet | 320  |  否    |    8    |   270e  |      -        |  27.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1.yml) |
| MobileNet-V3        | ImageNet | 608  |  否    |    8    |   270e  |      -        |  31.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v3.yml) |
| MobileNet-V3        | ImageNet | 416  |  否    |    8    |   270e  |      -        |  29.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v3.yml) |
| MobileNet-V3        | ImageNet | 320  |  否    |    8    |   270e  |      -        |  27.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v3.yml) |
| ResNet34            | ImageNet | 608  |  否    |    8    |   270e  |    63.356     |  36.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34.yml) |
| ResNet34            | ImageNet | 416  |  否    |    8    |   270e  |      -        |  34.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34.yml) |
| ResNet34            | ImageNet | 320  |  否    |    8    |   270e  |      -        |  31.4  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34.yml) |
| ResNet50_vd         | ImageNet | 608  |  是    |    8    |   270e  |      -        |  39.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/yolov3_r50vd_dcn.yml) |
| ResNet50_vd         | Object365 | 608  |  是    |    8    |   270e  |      -        |  41.4  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/yolov3_r50vd_dcn_obj365_pretrained_coco.yml) |

### YOLO v3 基于Pasacl VOC数据集

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 | 配置文件 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: |
| DarkNet53    | 608  |    8    |   270e  |    54.977     |  83.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet_voc.yml) |
| DarkNet53    | 416  |    8    |   270e  |      -        |  83.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet_voc.yml) |
| DarkNet53    | 320  |    8    |   270e  |      -        |  82.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet_voc.yml) |
| DarkNet53 Diou-Loss  | 608  |     8     |  270e   |       -        |  83.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc_diouloss.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_darknet_voc_diouloss.yml) |
| MobileNet-V1 | 608  |    8    |   270e  |   104.291     |  76.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1_voc.yml) |
| MobileNet-V1 | 416  |    8    |   270e  |      -        |  76.7  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1_voc.yml) |
| MobileNet-V1 | 320  |    8    |   270e  |      -        |  75.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_mobilenet_v1_voc.yml) |
| ResNet34     | 608  |    8    |   270e  |    82.247     |  82.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34_voc.yml) |
| ResNet34     | 416  |    8    |   270e  |      -        |  81.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34_voc.yml) |
| ResNet34     | 320  |    8    |   270e  |      -        |  80.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov3_r34_voc.yml) |

**注意事项:**

- 上表中也提供了原论文[YOLOv3](https://arxiv.org/abs/1804.02767)中YOLOv3-DarkNet53的精度，我们的实现版本主要从在bounding box的宽度和高度回归上使用了L1损失，图像mixup和label smooth等方法优化了其精度。
- YOLO v3在8卡，总batch size为64下训练270轮。数据增强包括：mixup, 随机颜色失真，随机剪裁，随机扩张，随机插值法，随机翻转。YOLO v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型测试不同尺寸图片，我们分别提供了尺寸为608/416/320大小的测试结果。deformable卷积作用在骨架网络5阶段。
- 在YOLOv3-DarkNet53模型基础上使用Diou-Loss后，在VOC数据集上该模型平均mAP比原模型高大约2%。
- YOLO v3增强版模型通过引入可变形卷积，dropblock，IoU loss和Iou aware，将精度进一步提升至43.6， 详情见[YOLOv3增强模型](./featured_model/YOLOv3_ENHANCEMENT.md)

### RetinaNet

|   骨架网络        | 每张GPU图片个数 | 学习率策略 | 推理时间(fps) | Box AP | 下载  | 配置文件 |
| :---------------: | :-----: | :-----: | :----: | :----: | :-------: | :----: |
| ResNet50-FPN      |    2    |   1x    | - | 36.0  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/retinanet_r50_fpn_1x.yml) |
| ResNet101-FPN     |    2    |   1x    | - | 37.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/retinanet_r101_fpn_1x.yml) |
| ResNeXt101-vd-FPN |    1    |   1x    | - | 40.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_x101_vd_64x4d_fpn_1x.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/retinanet_x101_vd_64x4d_fpn_1x.yml) |

**注意事项:** RetinaNet系列模型中，在总batch size为16下情况下，初始学习率改为0.01。

### EfficientDet

| 尺度              | 每张GPU图片个数 | 学习率策略 | Box AP | 下载      | 配置文件 |
| :---------------: | :-----:         | :-----:    | :----: | :-------: | :----:   |
| EfficientDet-D0   | 16              | 300 epochs | 33.8   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/efficientdet_d0.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/efficientdet_d0.yml) |

**注意事项:** 在总batch size为128(8x16)时，基础学习率改为0.16。

### SSDLite

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略|推理时间(fps) | Box AP | 下载 | 配置文件 |
| :----------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: |
| MobileNet_v1 | 300 | 64 | Cosine decay(40w) | - | 23.6 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssdlite_mobilenet_v1.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_mobilenet_v1.yml) |
| MobileNet_v3 small | 320 | 64 | Cosine decay(40w) | - | 16.2 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_mobilenet_v3_small.yml) |
| MobileNet_v3 large | 320 | 64 | Cosine decay(40w) | - | 23.3 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_mobilenet_v3_large.yml) |
| MobileNet_v3 small w/ FPN | 320 | 64 | Cosine decay(40w) | - | 18.9 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small_fpn.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_mobilenet_v3_small_fpn.yml) |
| MobileNet_v3 large w/ FPN | 320 | 64 | Cosine decay(40w) | - | 24.3 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large_fpn.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_mobilenet_v3_large_fpn.yml) |
| GhostNet | 320 | 64 | Cosine decay(40w) | - | 23.3 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_ghostnet.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssdlite_ghostnet.yml) |

**注意事项:** SSDLite模型使用学习率余弦衰减策略在8卡GPU下总batch size为512。

### SSD

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略|推理时间(fps) | Box AP | 下载 | 配置文件 |
| :----------: | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: |
| VGG16        | 300  |     8   |   40万  |    81.613     |  25.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_vgg16_300.yml) |
| VGG16        | 512  |     8   |   40万  |    46.007     |  29.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_vgg16_512.yml) |

**注意事项:** VGG-SSD在总batch size为32下训练40万轮。

### SSD 基于Pascal VOC数据集

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载  | 配置文件 |
| :----------- | :--: | :-----: | :-----: |  :------------: | :----: | :-------: | :----: |
| MobileNet v1 | 300  |    32   |   120e  |     159.543     | 73.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_mobilenet_v1_voc.yml) |
| VGG16        | 300  |     8   |   240e  |     117.279     | 77.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_vgg16_300_voc.yml) |
| VGG16        | 512  |     8   |   240e  |      65.975     | 80.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512_voc.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ssd/ssd_vgg16_512_voc.yml) |

**注意事项:** MobileNet-SSD在2卡，总batch size为64下训练120周期。VGG-SSD在总batch size为32下训练240周期。数据增强包括：随机颜色失真，随机剪裁，随机扩张，随机翻转。

### 人脸检测

详细请参考[人脸检测模型](featured_model/FACE_DETECTION.md)。


### 基于Open Images V5数据集的物体检测

详细请参考[Open Images V5数据集基线模型](featured_model/champion_model/OIDV5_BASELINE_MODEL.md)。

### Anchor Free系列模型

详细请参考[Anchor Free系列模型](featured_model/ANCHOR_FREE_DETECTION.md)。
