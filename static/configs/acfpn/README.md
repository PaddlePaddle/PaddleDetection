# Attention-guided Context Feature Pyramid Network for Object Detection

## Introduction

- Attention-guided Context Feature Pyramid Network for Object Detection: [https://arxiv.org/abs/2005.11475](https://arxiv.org/abs/2005.11475)

```
Cao J, Chen Q, Guo J, et al. Attention-guided Context Feature Pyramid Network for Object Detection[J]. arXiv preprint arXiv:2005.11475, 2020.
```


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-ACFPN         | Faster     |     2     |   1x    |     23.432     |  39.6  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_acfpn_1x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/acfpn/faster_rcnn_r50_vd_acfpn_1x.yml) |
