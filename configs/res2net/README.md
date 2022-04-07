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

| Backbone                | Type           | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :------------- | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :-----: |
| Res2Net50-FPN            | Faster         |     2     |   1x    |     -     |  40.6  |    -    | [model](https://paddledet.bj.bcebos.com/models/faster_rcnn_res2net50_vb_26w_4s_fpn_1x_coco.pdparams) |  [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/res2net/faster_rcnn_res2net50_vb_26w_4s_fpn_1x_coco.yml)  |
| Res2Net50-FPN            | Mask         |     2     |   2x    |     -     |  42.4  |    38.1    | [model](https://paddledet.bj.bcebos.com/models/mask_rcnn_res2net50_vb_26w_4s_fpn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/res2net/mask_rcnn_res2net50_vb_26w_4s_fpn_2x_coco.yml) |
| Res2Net50-vd-FPN            | Mask         |     2     |   2x    |     -     |  42.6  |    38.1    | [model](https://paddledet.bj.bcebos.com/models/mask_rcnn_res2net50_vd_26w_4s_fpn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/res2net/mask_rcnn_res2net50_vd_26w_4s_fpn_2x_coco.yml) |

Note: all the above models are trained with 8 gpus.
