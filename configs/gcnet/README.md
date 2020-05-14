# GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond

## Introduction

- GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
: [https://arxiv.org/abs/1904.11492](https://arxiv.org/abs/1904.11492)

```
@article{DBLP:journals/corr/abs-1904-11492,
  author    = {Yue Cao and
               Jiarui Xu and
               Stephen Lin and
               Fangyun Wei and
               Han Hu},
  title     = {GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  journal   = {CoRR},
  volume    = {abs/1904.11492},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.11492},
  archivePrefix = {arXiv},
  eprint    = {1904.11492},
  timestamp = {Tue, 09 Jul 2019 16:48:55 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1904-11492},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Model Zoo

| Backbone                | Type       |     Context| Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :-------------: |  :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| ResNet50-vd-FPN         | Mask       | GC(c3-c5, r16, add)  |     2     |   2x    |     15.31     |  41.4  |    36.8    | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_gcb_add_r16_2x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/gcnet/mask_rcnn_r50_vd_fpn_gcb_add_r16_2x.yml) |
| ResNet50-vd-FPN         | Mask       | GC(c3-c5, r16, mul)  |     2     |   2x    |     15.35     |  40.7  |    36.1    | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_gcb_mul_r16_2x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/gcnet/mask_rcnn_r50_vd_fpn_gcb_mul_r16_2x.yml) |
