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

| Backbone                | Type           | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :------------- | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| HRNetV2p_W18            | Faster         |     1     |   1x    |    -      |  36.8  |    -    | [model](https://paddledet.bj.bcebos.com/models/faster_rcnn_hrnetv2p_w18_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.yml) |
| HRNetV2p_W18            | Faster         |     1     |   2x    |    -      |  39.0  |    -    | [model](https://paddledet.bj.bcebos.com/models/faster_rcnn_hrnetv2p_w18_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco.yml) |
