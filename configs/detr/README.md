# DETR

## Introduction


DETR is an object detection model based on transformer. We reproduced the model of the paper.


## Model Zoo

| Backbone | Model | Images/GPU  | Inf time (fps) | Box AP | Config | Download |
|:------:|:--------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | DETR  | 4 | --- | 42.3 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/detr/detr_r50_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/detr_r50_1x_coco.pdparams) |

**Notes:**

- DETR is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- DETR uses 8GPU to train 500 epochs.

## Citations
```
@inproceedings{detr,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  booktitle = {ECCV},
  year      = {2020}
}
```
