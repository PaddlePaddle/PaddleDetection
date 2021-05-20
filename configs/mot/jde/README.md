English | [简体中文](README_cn.md)

# JDE (Joint Detection and Embedding)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

[JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding) is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network.
<div align="center">
  <img src="../../../docs/images/mot16_jde.gif" width=500 />
</div>

## Model Zoo

### JDE on MOT-16 training set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  73.2  |  69.3  | 1351  |  6591  | 21625 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  70.1  |  65.2  | 1328  |  6441  | 25187 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.2  |  64.5  | 1308  |  7011  | 32252 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_576x320.yml) |


**Notes:**
 JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training JDE on 8 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./jde_darknet53_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml
```

### 2. Evaluation

Evaluating the track performance of JDE on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=output/jde_darknet53_30e_1088x608/model_final.pdparams
```

## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
