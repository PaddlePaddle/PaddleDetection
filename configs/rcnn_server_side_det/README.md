# Practical Server-side detection method base on RCNN

## Introduction

* This is developed by PaddleDetection. Many useful tricks are utilized for the model training process. More details can be seen in the configuration file.


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-FPN-Dcnv2         | Faster     |     2     |   3x    |     61.425     |  41.6  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet50-vd-FPN-Dcnv2         | Cascade Faster     |     2     |   3x    |     20.001     |  47.8  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
