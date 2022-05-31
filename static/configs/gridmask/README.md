# GridMask Data Augmentation

## Introduction

- GridMask Data Augmentation
: [https://arxiv.org/abs/2001.04086](https://arxiv.org/abs/2001.04086)

```
@article{chen2020gridmask,
  title={GridMask data augmentation},
  author={Chen, Pengguang},
  journal={arXiv preprint arXiv:2001.04086},
  year={2020}
}
```


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-FPN         | Faster     |     2     |   4x    |     21.847     |  39.1%  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_gridmask_4x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/gridmask/faster_rcnn_r50_vd_fpn_gridmask_4x.yml) |
