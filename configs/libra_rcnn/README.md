# Libra R-CNN: Towards Balanced Learning for Object Detection

## Introduction

- Libra R-CNN: Towards Balanced Learning for Object Detection
: [https://arxiv.org/abs/1904.02701](https://arxiv.org/abs/1904.02701)

```
@inproceedings{pang2019libra,
  title={Libra R-CNN: Towards Balanced Learning for Object Detection},
  author={Pang, Jiangmiao and Chen, Kai and Shi, Jianping and Feng, Huajun and Ouyang, Wanli and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-BFP         | Faster     |     2     |   1x    |     18.247     |  40.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/libra_rcnn_r50_vd_fpn_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/libra_rcnn/libra_rcnn_r50_vd_fpn_1x.yml) |
| ResNet101-vd-BFP         | Faster     |     2     |   1x    |     14.865     |  42.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/libra_rcnn_r101_vd_fpn_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/libra_rcnn/libra_rcnn_r101_vd_fpn_1x.yml) |
