# Practical Mobile-side detection method base on RCNN

## Introduction

* This is developed by PaddleDetection. Many useful tricks are utilized for the model training process. More details can be seen in the configuration file.
* The inerence is tested on Qualcomm Snapdragon 845 Mobile Platform.


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time on SD845 (fps) | Box AP | Mask AP |                           Download                           |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: |
| MobileNetV3-vd-FPN         | Cascade Faster     |     2     |   5.6x(CosineDecay)    |     8.13     |  25.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_mobilenetv3_fpn_320.tar) |
| MobileNetV3-vd-FPN         | Cascade Faster     |     2     |   5.6x(CosineDecay)    |     2.66     |  30.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_mobilenetv3_fpn_640.tar) |

**note**
* `5.6x` means the model is trained with `50000` minibatches 8 GPU cards(batch size=2 for each card).
