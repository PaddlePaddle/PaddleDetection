# Hybrid Task Cascade for Instance Segmentation

## Introduction

We provide config files to reproduce the results in the CVPR 2019 paper for [Hybrid Task Cascade](https://arxiv.org/abs/1901.07518).

```
@inproceedings{chen2019hybrid,
    title={Hybrid task cascade for instance segmentation},
      author={Chen, Kai and Pang, Jiangmiao and Wang, Jiaqi and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Shi, Jianping and Ouyang, Wanli and Chen Change Loy and Dahua Lin},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          year={2019}
}
```

## Dataset

HTC requires COCO and COCO-stuff dataset for training.

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

  | Backbone  | Lr schd | Inf time (fps) | box AP | mask AP | Download |
  |:---------:|:-------:|:--------------:|:------:|:-------:|:--------:|
  | R-50-FPN  | 1x      | 11             | 42.9   | 37.0    | [model](https://paddlemodels.bj.bcebos.com/object_detection/htc_r50_fpn_1x.pdparams ) |
