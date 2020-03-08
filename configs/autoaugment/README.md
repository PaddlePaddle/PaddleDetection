# Learning Data Augmentation Strategies for Object Detection

## Introduction

- Learning Data Augmentation Strategies for Object Detection: [https://arxiv.org/abs/1906.11172](https://arxiv.org/abs/1906.11172)

```
@article{Zoph2019LearningDA,
  title={Learning Data Augmentation Strategies for Object Detection},
  author={Barret Zoph and Ekin Dogus Cubuk and Golnaz Ghiasi and Tsung-Yi Lin and Jonathon Shlens and Quoc V. Le},
  journal={ArXiv},
  year={2019},
  volume={abs/1906.11172}
}
```


## Model Zoo

| Backbone                | Type     | AutoAug policy | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           |
| :---------------------- | :-------------:| :-------: | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50-vd-FPN         | Faster     |   v1 |  2     |   3x    |     22.800     |  39.9  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_aa_3x.tar) |
| ResNet101-vd-FPN         | Faster     |   v1 |  2     |   3x    |     17.652     |  42.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_aa_3x.tar) |
