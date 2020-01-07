English | [简体中文](MODEL_ZOO_cn.md)

# Model Zoo and Benchmark
## Environment

- Python 2.7.1
- PaddlePaddle >=1.5
- CUDA 9.0
- cuDNN >=7.4
- NCCL 2.1.2

## Common settings

- All models below were trained on `coco_2017_train`, and tested on `coco_2017_val`.
- Batch Normalization layers in backbones are replaced by Affine Channel layers.
- Unless otherwise noted, all ResNet backbones adopt the [ResNet-B](https://arxiv.org/pdf/1812.01187) variant..
- For RCNN and RetinaNet models, only horizontal flipping data augmentation was used in the training phase and no augmentations were used in the testing phase.
- **Inf time (fps)**: the inference time is measured with fps (image/s) on a single GPU (Tesla V100) with cuDNN 7.5 by running 'tools/eval.py' on all validation set, which including data loadding, network forward and post processing. The batch size is 1.


## Training Schedules

- We adopt exactly the same training schedules as [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules).
- 1x indicates the schedule starts at a LR of 0.02 and is decreased by a factor of 10 after 60k and 80k iterations and eventually terminates at 90k iterations for minibatch size 16. For batch size 8, LR is decreased to 0.01, total training iterations are doubled, and the decay milestones are scaled by 2.
- 2x schedule is twice as long as 1x, with the LR milestones scaled accordingly.

## ImageNet Pretrained Models

The backbone models pretrained on ImageNet are available. All backbone models are pretrained on standard ImageNet-1k dataset and can be downloaded [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances).

- **Notes:**  The ResNet50 model was trained with cosine LR decay schedule and can be downloaded [here](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar).

## Baselines

### Faster & Mask R-CNN

| Backbone                | Type           | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           |
| :---------------------- | :------------- | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50                | Faster         |     1     |   1x    |     12.747     |  35.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) |
| ResNet50                | Faster         |     1     |   2x    |     12.686     |  37.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) |
| ResNet50                | Mask           |     1     |   1x    |     11.615     |  36.5  |  32.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) |
| ResNet50                | Mask           |     1     |   2x    |     11.494     |  38.2  |  33.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) |
| ResNet50-vd             | Faster         |     1     |   1x    |     12.575     |  36.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN            | Faster         |     2     |   1x    |     22.273     |  37.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN            | Faster         |     2     |   2x    |     22.297     |  37.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN            | Mask           |     1     |   1x    |     15.184     |  37.9  |  34.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN            | Mask           |     1     |   2x    |     15.881     |  38.7  |  34.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN            | Cascade Faster |     2     |   1x    |     17.507     |  40.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN            | Cascade Mask   |     1     |   1x    |       -        |  41.3  |  35.5   | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN         | Faster         |     2     |   2x    |     21.847     |  38.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) |
| ResNet50-vd-FPN         | Mask           |     1     |   2x    |     15.825     |  39.8  |  35.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) |
| CBResNet50-vd-FPN         | Faster         |     2     |   1x    |     -     |  39.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_cbr50_vd_dual_fpn_1x.tar) |
| ResNet101               | Faster         |     1     |   1x    |     9.316      |  38.3  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) |
| ResNet101-FPN           | Faster         |     1     |   1x    |     17.297     |  38.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-FPN           | Faster         |     1     |   2x    |     17.246     |  39.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| ResNet101-FPN           | Mask           |     1     |   1x    |     12.983     |  39.5  |  35.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) |
| ResNet101-vd-FPN        | Faster         |     1     |   1x    |     17.011     |  40.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_1x.tar) |
| ResNet101-vd-FPN        | Faster         |     1     |   2x    |     16.934     |  40.8  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |
| ResNet101-vd-FPN        | Mask           |     1     |   1x    |     13.105     |  41.4  |  36.8   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) |
| CBResNet101-vd-FPN         | Faster         |     2     |   1x    |     -     |  42.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_cbr101_vd_dual_fpn_1x.tar) |
| ResNeXt101-vd-64x4d-FPN | Faster         |     1     |   1x    |     8.815      |  42.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-64x4d-FPN | Faster         |     1     |   2x    |     8.809      |  41.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| ResNeXt101-vd-64x4d-FPN | Mask           |     1     |   1x    |     7.689      |  42.9  |  37.9   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-64x4d-FPN | Mask           |     1     |   2x    |     7.859      |  42.6  |  37.6   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| SENet154-vd-FPN         | Faster         |     1     |  1.44x  |     3.408      |  42.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_vd_fpn_s1x.tar) |
| SENet154-vd-FPN         | Mask           |     1     |  1.44x  |     3.233      |  44.0  |  38.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |
| ResNet101-vd-FPN            | CascadeClsAware Faster   |     2     |   1x    |     -     |  44.7(softnms)  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r101_vd_fpn_1x_softnms.tar) |

### Deformable ConvNets v2

| Backbone                | Type           | Conv  | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           |
| :---------------------- | :------------- | :---: | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN            | Faster         | c3-c5 |     2     |   1x    |     19.978     |  41.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN         | Faster         | c3-c5 |     2     |   2x    |     19.222     |  42.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN        | Faster         | c3-c5 |     2     |   1x    |     14.477     |  44.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-64x4d-FPN | Faster         | c3-c5 |     1     |   1x    |     7.209      |  45.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN            | Mask           | c3-c5 |     1     |   1x    |     14.53      |  41.9  |  37.3   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN         | Mask           | c3-c5 |     1     |   2x    |     14.832     |  42.9  |  38.0   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN        | Mask           | c3-c5 |     1     |   1x    |     11.546     |  44.6  |  39.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-64x4d-FPN | Mask           | c3-c5 |     1     |   1x    |      6.45      |  46.2  |  40.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN            | Cascade Faster | c3-c5 |     2     |   1x    |       -        |  44.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet101-vd-FPN        | Cascade Faster | c3-c5 |     2     |   1x    |       -        |  46.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN       | Cascade Faster | c3-c5 |     2     |   1x    |       -        |  47.3  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| SENet154-vd-FPN         | Cascade Mask   | c3-c5 |    1      |  1.44x  |       -        |  51.9  |  43.9   | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_dcnv2_se154_vd_fpn_gn_s1x.tar) |
| ResNet200-vd-FPN-Nonlocal    | CascadeClsAware Faster  | c3-c5 |     1     |   2.5x    |     -     |  51.7%(softnms)  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.tar) |
| CBResNet200-vd-FPN-Nonlocal | Cascade Faster  | c3-c5 |     1     |   2.5x    |     -     |  53.3%(softnms)  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar) |


#### Notes:
- Deformable ConvNets v2(dcn_v2) reference from [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5` means adding `dcn` in resnet stage 3 to 5.
- Detailed configuration file in [configs/dcn](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn)


### HRNet
* See more details in [HRNet model zoo](../configs/hrnet/README.md).


### Res2Net
* See more details in [Res2Net model zoo](../configs/res2net/README.md).

### IOU loss
* GIOU loss and DIOU loss are included now. See more details in [IOU loss model zoo](../configs/iou_loss/README.md).

### GCNet
* See more details in [GCNet model zoo](../configs/gcnet/README.md).


### Group Normalization
| Backbone             | Type           | Image/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN         | Faster         |    2    |   2x    |  39.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_gn_2x.tar) |
| ResNet50-FPN         | Mask           |    1    |   2x    |  40.1  |   35.8  | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_gn_2x.tar) |

#### Notes:
- Group Normalization reference from [Group Normalization](https://arxiv.org/abs/1803.08494).
- Detailed configuration file in [configs/gn](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/gn)

### YOLO v3

| Backbone     | Pretrain dataset | Size | deformable Conv | Image/gpu | Lr schd | Inf time (fps) | Box AP |  Download |
| :----------- | :--------: | :-----: | :-----: |:------------: |:----: | :-------: | :----: | :-------: |
| DarkNet53 (paper) | ImageNet | 608  |  False    |    8    |   270e  |      -        |  33.0  | - |
| DarkNet53 (paper) | ImageNet | 416  |  False    |    8    |   270e  |      -        |  31.0  | - |
| DarkNet53 (paper) | ImageNet | 320  |  False    |    8    |   270e  |      -        |  28.2  | - |
| DarkNet53         | ImageNet | 608  |  False    |    8    |   270e  |    45.571     |  38.9  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53         | ImageNet | 416  |  False    |    8    |   270e  |      -        |  37.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53         | ImageNet | 320  |  False    |    8    |   270e  |      -        |  34.8  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| MobileNet-V1      | ImageNet | 608  |  False    |    8    |   270e  |    78.302     |  29.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1      | ImageNet | 416  |  False    |    8    |   270e  |      -        |  29.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1      | ImageNet | 320  |  False    |    8    |   270e  |      -        |  27.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| ResNet34          | ImageNet | 608  |  False    |    8    |   270e  |    63.356     |  36.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34          | ImageNet | 416  |  False    |    8    |   270e  |      -        |  34.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34          | ImageNet | 320  |  False    |    8    |   270e  |      -        |  31.4  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet50_vd       | ImageNet | 608  |  True     |    8    |   270e  |      -        |  39.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) |
| ResNet50_vd       | Object365 | 608  |  True    |    8    |   270e  |      -        |  41.4  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |

### YOLO v3 on Pascal VOC

| Backbone     | Size | Image/gpu | Lr schd | Inf time (fps) | Box AP |                           Download                           |
| :----------- | :--: | :-------: | :-----: | :------------: | :----: | :----------------------------------------------------------: |
| DarkNet53    | 608  |     8     |  270e   |     54.977     |  83.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 416  |     8     |  270e   |       -        |  83.6  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 320  |     8     |  270e   |       -        |  82.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| MobileNet-V1 | 608  |     8     |  270e   |    104.291     |  76.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 416  |     8     |  270e   |       -        |  76.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 320  |     8     |  270e   |       -        |  75.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| ResNet34     | 608  |     8     |  270e   |     82.247     |  82.6  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 416  |     8     |  270e   |       -        |  81.9  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 320  |     8     |  270e   |       -        |  80.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |

#### Notes:
- YOLOv3-DarkNet53 performance in paper [YOLOv3](https://arxiv.org/abs/1804.02767) is also provided above, our implements
improved performance mainly by using L1 loss in bounding box width and height regression, image mixup and label smooth.
- YOLO v3 is trained in 8 GPU with total batch size as 64 and trained 270 epoches. YOLO v3 training data augmentations: mixup,
randomly color distortion, randomly cropping, randomly expansion, randomly interpolation method, randomly flippling. YOLO v3 used randomly
reshaped minibatch in training, inferences can be performed on different image sizes with the same model weights, and we provided evaluation
results of image size 608/416/320 above. Deformable conv is added on stage 5 of backbone.

### RetinaNet

| Backbone          | Image/gpu | Lr schd | Box AP | Download  |
| :---------------: | :-----: | :-----: | :----: | :-------: |
| ResNet50-FPN      |    2    |   1x    |  36.0  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar)  |
| ResNet101-FPN     |    2    |   1x    |  37.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) |
| ResNeXt101-vd-FPN |    1    |   1x    |  40.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_x101_vd_64x4d_fpn_1x.tar) |

**Notes:** In RetinaNet, the base LR is changed to 0.01 for minibatch size 16.

### SSD

| Backbone | Size | Image/gpu | Lr schd | Inf time (fps) | Box AP |                           Download                           |
| :------: | :--: | :-------: | :-----: | :------------: | :----: | :----------------------------------------------------------: |
|  VGG16   | 300  |     8     |   40w   |     81.613     |  25.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar) |
|  VGG16   | 512  |     8     |   40w   |     46.007     |  29.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512.tar) |

**Notes:** VGG-SSD is trained in 4 GPU with total batch size as 32 and trained 400000 iters.

### SSD on Pascal VOC

| Backbone     | Size | Image/gpu | Lr schd | Inf time (fps) | Box AP |                           Download                           |
| :----------- | :--: | :-------: | :-----: | :------------: | :----: | :----------------------------------------------------------: |
| MobileNet v1 | 300  |    32     |  120e   |    159.543     |  73.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) |
| VGG16        | 300  |     8     |  240e   |    117.279     |  77.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300_voc.tar) |
| VGG16        | 512  |     8     |  240e   |     65.975     |  80.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512_voc.tar) |

**NOTE**: MobileNet-SSD is trained in 2 GPU with totoal batch size as 64 and trained 120 epoches. VGG-SSD is trained in 4 GPU with total batch size as 32 and trained 240 epoches. SSD training data augmentations: randomly color distortion,
randomly cropping, randomly expansion, randomly flipping.


## Face Detection

Please refer [face detection models](../configs/face_detection) for details.


## Object Detection in Open Images Dataset V5

Please refer [Open Images Dataset V5 Baseline model](OIDV5_BASELINE_MODEL.md) for details.
