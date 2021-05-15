# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

FairMOT focuses on accomplishing the detection and re-identification in a single network to improve the inference speed, presents a simple baseline which consists of two homogeneous branches to predict pixel-wise objectness scores and re-ID features. The achieved fairness between the two tasks allows FairMOT to obtain high levels of detection and tracking accuracy.


## Model Zoo

### Results on MOT-16 train set

| backbone      | input shape  | MOTA   | IDF1   |  IDS  |   FP  |   FN  | download  | config |
| :-----------------| :------- | :----: | :----: | :---: | :----: | :---: |:---: | :---: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  | 544  |  3822  | 14095 | ---- | ---- |
| DLA-34         | 1088x608 |  83.4  |  82.7  | 517  | 4077   | 13761 |  [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |


### Results on MOT-16 test set

| backbone      | input shape  | MOTA   | IDF1   |  IDS  |   MT  |   ML  | download  | config |
| :-----------------| :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: |
| DLA-34(paper)  | 1088x608 |    74.9   72.8    1074    44.7%   15.9%  | ---- | ---- |
| DLA-34         | 1088x608 |  74.7  |  72.8  | 1044  | 41.9%   | 19.1% |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

**Notes:**

FairMOT used 2 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training FairMOT on 2 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml &>fairmot_dla34_30e_1088x608.log 2>&1 &
```


### 2. Evaluation

Evaluating the track performance of FairMOT on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final
```

## Citations
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```
