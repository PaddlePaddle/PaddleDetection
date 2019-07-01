# Model Zoo and Benchmark
## Environment

- Python 2.7.1
- PaddlePaddle 1.5
- CUDA 9.0
- CUDNN 7.4
- NCCL 2.1.2

## Common settings

- All models below except SSD were trained on `coco_2017_train`, and tested on `coco_2017_val`.
- Batch Normalization layers in backbones are replaced by Affine Channel layers.
- Unless otherwise noted, all ResNet backbones adopt the [ResNet-B](https://arxiv.org/pdf/1812.01187) variant..
- For RCNN and RetinaNet models, only horizontal flipping data augmentation was used in the training phase and no augmentations were used in the testing phase.

## Training Schedules

- We adopt exactly the same training schedules as [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules).
- 1x indicates the schedule starts at a LR of 0.02 and is decreased by a factor of 10 after 60k and 80k iterations and eventually terminates at 90k iterations for minibatch size 16. For batch size 8, LR is decreased to 0.01, total training iterations are doubled, and the decay milestones are scaled by 2.
- 2x schedule is twice as long as 1x, with the LR milestones scaled accordingly.

## ImageNet Pretrained Models

The backbone models pretrained on ImageNet are available. All backbone models are pretrained on standard ImageNet-1k dataset and can be downloaded [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances).

- Notes:  The ResNet50 model was trained with cosine LR decay schedule and can be downloaded here [here](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar).

## Baselines

### Faster & Mask R-CNN

| Backbone             | Type           | Img/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50             | Faster         |    1    |   1x    |  35.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) |
| ResNet50             | Faster         |    1    |   2x    |  37.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) |
| ResNet50             | Mask           |    1    |   1x    |  36.5  |  32.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) |
| ResNet50             | Mask           |    1    |   2x    |        |         |                          [model]()                           |
| ResNet50-D           | Faster         |    1    |   1x    |  36.4  |    -    | [model](ttps://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   1x    |  37.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   2x    |  37.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Mask           |    2    |   1x    |  37.9  |  34.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |  40.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) |
| ResNet50-D-FPN       | Faster         |    2    |   2x    |  38.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) |
| ResNet50-D-FPN       | Mask           |    2    |   2x    |  39.8  |  35.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) |
| ResNet101            | Faster         |    1    |   1x    |  38.3  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   1x    |  38.7  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   2x    |  39.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| ResNet101-FPN        | Mask           |    1    |   1x    |  39.5  |  35.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) |
| ResNet101-D-FPN      | Faster         |    1    |   1x    |  40.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-D-FPN      | Faster         |    1    |   2x    |  40.6  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| SENet154-D-FPN       | Faster         |    1    |  1.44x  |  43.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_fpn_s1x.tar) |
| SENet154-D-FPN       | Mask           |    1    |  1.44x  |  44.0  |  38.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |

### Yolo v3

| Backbone  | Size | Lr schd | Box AP | Download  |
| :-------- | :--: | :-----: | :----: | :-------: |
| DarkNet53 | 608  |  120e   |  25.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| MobileNet-V1 | 608  |  120e   |  25.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| ResNet34 | 608  |  120e   |  25.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |

- Notes: Data Augmentation（TODO：Kaipeng）

### RetinaNet

| Backbone     | Size | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :----: | :-------: |
| ResNet50-FPN | 300  |  120e   |  36.0  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar) |
| ResNet101-FPN | 300  |  120e   |  37.3  | [model](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) |

- Notes: （TODO：Kaipeng）

### SSD on PascalVOC

| Backbone     | Size | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :----: | :-------: |
| MobileNet v1 | 300  |  120e   |  25.7  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) |
