# Res2Net

## Introduction

- Res2Net: A New Multi-scale Backbone Architecture: [https://arxiv.org/abs/1904.01169](https://arxiv.org/abs/1904.01169)

```
@article{DBLP:journals/corr/abs-1904-01169,
  author    = {Shanghua Gao and
               Ming{-}Ming Cheng and
               Kai Zhao and
               Xinyu Zhang and
               Ming{-}Hsuan Yang and
               Philip H. S. Torr},
  title     = {Res2Net: {A} New Multi-scale Backbone Architecture},
  journal   = {CoRR},
  volume    = {abs/1904.01169},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.01169},
  archivePrefix = {arXiv},
  eprint    = {1904.01169},
  timestamp = {Thu, 25 Apr 2019 10:24:54 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1904-01169},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Model Zoo

| Backbone                | Type           | deformable Conv  | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :------------- | :---: | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| Res2Net50-FPN            | Faster         | False |     2     |   1x    |     20.320     |  39.5  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_res2net50_vb_26w_4s_fpn_1x.tar) |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/res2net/faster_rcnn_res2net50_vb_26w_4s_fpn_1x.yml) |
| Res2Net50-FPN            | Mask         | False |     2     |   2x    |     16.069     |  40.7  |    36.2    | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_res2net50_vb_26w_4s_fpn_2x.tar) |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/res2net/faster_rcnn_res2net50_vb_26w_4s_fpn_2x.yml) |
| Res2Net50-vd-FPN            | Mask         | False |     2     |   2x    |     15.816     |  40.9  |    36.2    | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_res2net50_vd_26w_4s_fpn_2x.tar) |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/res2net/mask_rcnn_res2net50_vd_26w_4s_fpn_2x.yml) |
| Res2Net50-vd-FPN            | Mask         | True |     2     |   2x    |     14.478     |  43.5  |    38.4    | [model](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_res2net50_vd_26w_4s_fpn_dcnv2_1x.tar) |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/res2net/mask_rcnn_res2net50_vd_26w_4s_fpn_dcnv2_1x.yml) |
