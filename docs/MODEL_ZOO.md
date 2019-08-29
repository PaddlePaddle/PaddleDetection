# Model Zoo and Benchmark
## Environment

- Python 2.7.1
- PaddlePaddle 1.5
- CUDA 9.0
- CUDNN 7.4
- NCCL 2.1.2

## Common settings

- All models below were trained on `coco_2017_train`, and tested on `coco_2017_val`.
- Batch Normalization layers in backbones are replaced by Affine Channel layers.
- Unless otherwise noted, all ResNet backbones adopt the [ResNet-B](https://arxiv.org/pdf/1812.01187) variant..
- For RCNN and RetinaNet models, only horizontal flipping data augmentation was used in the training phase and no augmentations were used in the testing phase.

## Training Schedules

- We adopt exactly the same training schedules as [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules).
- 1x indicates the schedule starts at a LR of 0.02 and is decreased by a factor of 10 after 60k and 80k iterations and eventually terminates at 90k iterations for minibatch size 16. For batch size 8, LR is decreased to 0.01, total training iterations are doubled, and the decay milestones are scaled by 2.
- 2x schedule is twice as long as 1x, with the LR milestones scaled accordingly.

## ImageNet Pretrained Models

The backbone models pretrained on ImageNet are available. All backbone models are pretrained on standard ImageNet-1k dataset and can be downloaded [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances).

- **Notes:**  The ResNet50 model was trained with cosine LR decay schedule and can be downloaded [here](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar).

## Baselines

### Faster & Mask R-CNN

| Backbone             | Type           | Image/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50             | Faster         |    1    |   1x    |  35.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) |
| ResNet50             | Faster         |    1    |   2x    |  37.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) |
| ResNet50             | Mask           |    1    |   1x    |  36.5  |  32.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) |
| ResNet50             | Mask           |    1    |   2x    |  38.2  |  33.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) |
| ResNet50-vd          | Faster         |    1    |   1x    |  36.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   1x    |  37.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   2x    |  37.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Mask           |    1    |   1x    |  37.9  |  34.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Mask           |    1    |   2x    |  38.7  |  34.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |  40.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Cascade Mask   |    1    |   1x    |  41.3  |  35.5   | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Faster         |    2    |   2x    |  38.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) |
| ResNet50-vd-FPN      | Mask           |    1    |   2x    |  39.8  |  35.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) |
| ResNet101            | Faster         |    1    |   1x    |  38.3  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   1x    |  38.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   2x    |  39.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| ResNet101-FPN        | Mask           |    1    |   1x    |  39.5  |  35.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |  40.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |  40.8  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Mask           |    1    |   1x    |  41.4  |  36.8   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |  42.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |  41.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   1x    |  42.9  |  37.9   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   2x    |  42.6  |  37.6   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| SENet154-vd-FPN      | Faster         |    1    |  1.44x  |  42.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_vd_fpn_s1x.tar) |
| SENet154-vd-FPN      | Mask           |    1    |  1.44x  |  44.0  |  38.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |

### Deformable ConvNets v2

| Backbone             | Type           | Conv    | Image/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: |:--------: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN         | Faster         | c3-c5   |    2      |   1x    |  41.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Faster         | c3-c5   |    2      |   2x    |  42.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Faster         | c3-c5   |    2      |   1x    |  44.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         | c3-c5   |    1      |   1x    |  45.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN         | Mask           | c3-c5   |    1      |   1x    |  41.9  |  37.3   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Mask           | c3-c5   |    1      |   2x    |  42.9  |  38.0   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r50_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Mask           | c3-c5   |    1      |   1x    |  44.6  |  39.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Mask           | c3-c5   |    1      |   1x    |  46.2  |  40.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |
| ResNet50-FPN         | Cascade Faster | c3-c5   |    2      |   1x    |  44.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_fpn_1x.tar) |
| ResNet101-vd-FPN     | Cascade Faster | c3-c5   |    2      |   1x    |  46.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Cascade Faster | c3-c5   |    2      |   1x    |  47.3  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x.tar) |

#### Notes:
- Deformable ConvNets v2(dcn_v2) reference from [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5` means adding `dcn` in resnet stage 3 to 5.
- Detailed configuration file in [configs/dcn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection/configs/dcn)

### Group Normalization
| Backbone             | Type           | Image/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-FPN         | Faster         |    2    |   2x    |  39.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_gn_2x.tar) |
| ResNet50-FPN         | Mask           |    1    |   2x    |  40.1  |   35.8  | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_gn_2x.tar) |

#### Notes:
- Group Normalization reference from [Group Normalization](https://arxiv.org/abs/1803.08494).
- Detailed configuration file in [configs/gn](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection/configs/gn)

### Yolo v3

| Backbone     | Size | Image/gpu | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |  38.9  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 416  |    8    |   270e  |  37.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 320  |    8    |   270e  |  34.8  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |  29.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |  29.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |  27.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| ResNet34     | 608  |    8    |   270e  |  36.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 416  |    8    |   270e  |  34.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 320  |    8    |   270e  |  31.4  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |

### Yolo v3 on Pascal VOC

| Backbone     | Size | Image/gpu | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |  83.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 416  |    8    |   270e  |  83.6  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 320  |    8    |   270e  |  82.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |  76.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |  76.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |  75.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| ResNet34     | 608  |    8    |   270e  |  82.6  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 416  |    8    |   270e  |  81.9  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 320  |    8    |   270e  |  80.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |

**Notes:** Yolo v3 is trained in 8 GPU with total batch size as 64 and trained 270 epoches. Yolo v3 training data augmentations: mixup,
randomly color distortion, randomly cropping, randomly expansion, randomly interpolation method, randomly flippling. Yolo v3 used randomly
reshaped minibatch in training, inferences can be performed on different image sizes with the same model weights, and we provided evaluation
results of image size 608/416/320 above.

### RetinaNet

| Backbone          | Image/gpu | Lr schd | Box AP | Download  |
| :---------------: | :-----: | :-----: | :----: | :-------: |
| ResNet50-FPN      |    2    |   1x    |  36.0  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar)  |
| ResNet101-FPN     |    2    |   1x    |  37.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) |
| ResNeXt101-vd-FPN |    1    |   1x    |  40.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_x101_vd_64x4d_fpn_1x.tar) |

**Notes:** In RetinaNet, the base LR is changed to 0.01 for minibatch size 16.

### SSD

| Backbone     | Size | Image/gpu | Lr schd | Box AP | Download  |
| :----------: | :--: | :-------: | :-----: | :----: | :-------: |
| VGG16        | 300  |     8   |   40w  |  25.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300.tar) |
| VGG16        | 512  |     8   |   40w  |  29.1  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512.tar) |

**Notes:** VGG-SSD is trained in 4 GPU with total batch size as 32 and trained 400000 iters.

### SSD on Pascal VOC

| Backbone     | Size | Image/gpu | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| MobileNet v1 | 300  |    32   |   120e  |  73.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) |
| VGG16        | 300  |     8   |   240e  |  77.5  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_300_voc.tar) |
| VGG16        | 512  |     8   |   240e  |  80.2  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_vgg16_512_voc.tar) |

**NOTE**: MobileNet-SSD is trained in 2 GPU with totoal batch size as 64 and trained 120 epoches. VGG-SSD is trained in 4 GPU with total batch size as 32 and trained 240 epoches. SSD training data augmentations: randomly color distortion,
randomly cropping, randomly expansion, randomly flipping.
