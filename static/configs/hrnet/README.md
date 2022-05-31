# High-resolution networks (HRNets) for object detection

## Introduction

- Deep High-Resolution Representation Learning for Human Pose Estimation: [https://arxiv.org/abs/1902.09212](https://arxiv.org/abs/1902.09212)

```
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}
```

- High-Resolution Representations for Labeling Pixels and Regions: [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514)

```
@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Model Zoo

| Backbone                | Type           | deformable Conv  | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :------------- | :---: | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| HRNetV2p_W18            | Faster         | False |     2     |   1x    |     17.509     |  36.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.yml) |
| HRNetV2p_W18            | Faster         | False |     2     |   2x    |     17.509     |  38.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_2x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/hrnet/faster_rcnn_hrnetv2p_w18_2x.yml) |
