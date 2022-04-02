# TOOD

## Introduction

[TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755)

TOOD is an object detection model. We reproduced the model of the paper.


## Model Zoo

| Backbone | Model | Images/GPU  | Inf time (fps) | Box AP | Config | Download |
|:------:|:--------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | TOOD  | 4 | --- | 42.5 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/tood/tood_r50_fpn_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/tood_r50_fpn_1x_coco.pdparams) |

**Notes:**

- TOOD is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- TOOD uses 8GPU to train 12 epochs.

GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/tood/tood_r50_fpn_1x_coco.yml --fleet
```

## Citations
```
@inproceedings{feng2021tood,
    title={TOOD: Task-aligned One-stage Object Detection},
    author={Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R and Huang, Weilin},
    booktitle={ICCV},
    year={2021}
}
```
