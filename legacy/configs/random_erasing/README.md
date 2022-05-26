# Random Erasing Data Augmentation

## Introduction

- Random Erasing Data Augmentation
: [https://arxiv.org/abs/1708.04896](https://arxiv.org/abs/1708.04896)

```
@article{zhong1708random,
  title={Random erasing data augmentation. arXiv 2017},
  author={Zhong, Z and Zheng, L and Kang, G and Li, S and Yang, Y},
  journal={arXiv preprint arXiv:1708.04896}
}
```


## Model Zoo

| Backbone                | Type     | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-FPN         | Faster     |     2     |   4x    |     21.847     |  39.0%  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_random_erasing_4x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/random_erasing/faster_rcnn_r50_vd_fpn_random_erasing_4x.yml) |
