English | [简体中文](README_cn.md)

# CenterNet (CenterNet: Objects as Points)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Citations](#Citations)

## Introduction

[CenterNet](http://arxiv.org/abs/1904.07850) is an Anchor Free detector, which model an object as a single point -- the center point of its bounding box. The detector uses keypoint estimation to find center points and regresses to all other object properties. The center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors.

## Model Zoo

### CenterNet Results on COCO-val 2017

| backbone       | input shape | mAP   |    FPS    | download | config |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 512x512 |  37.4  |     -   |    -   |   -    |
| DLA-34         | 512x512 |  37.6  |     -   | [model](https://bj.bcebos.com/v1/paddledet/models/centernet_dla34_140e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/centernet/centernet_dla34_140e_coco.yml) |
| ResNet50 + DLAUp  | 512x512 |  38.9  |     -   | [model](https://bj.bcebos.com/v1/paddledet/models/centernet_r50_140e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/centernet/centernet_r50_140e_coco.yml) |

## Citations
```
@article{zhou2019objects,
  title={Objects as points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```
