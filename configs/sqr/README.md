# Enhanced Training of Query-Based Object Detection via Selective Query Recollection


## Introduction
This paper investigates a phenomenon where query-based object detectors mispredict at the last decoding stage while predicting correctly at an intermediate stage. It design and present Selective Query Recollection (SQR), a simple and effective training strategy for query-based object detectors. It cumulatively collects intermediate queries as decoding stages go deeper and selectively forwards the queries to the downstream stages aside from the sequential structure.


## Model Zoo

| Backbone |      Model          | Images/GPU | GPUs | Epochs | Box AP |            Config                                | Download  |
|:--------:|:-------------------:|:----------:|:----:|:------:|:------:|:------------------------------------------------:|:---------:|
|   R-50   | Deformable DETR SQR |     1      |  4   |   12   |  32.9  | [config](./deformable_detr_sqr_r50_12e_coco.yml) |[model](https://bj.bcebos.com/v1/paddledet/models/deformable_detr_sqr_r50_12e_coco.pdparams) |

> We did not find the config for the 12 epochs experiment in the paper, which we wrote ourselves with reference to the standard 12 epochs config in mmdetection. The same accuracy was obtained in the official project and in this project with this [config](./deformable_detr_sqr_r50_12e_coco.yml). <br> We haven't finished validating the 50 epochs experiment yet, if you need the config, please refer to [here](https://pan.baidu.com/s/1eWavnAiRoFXm3mMlpn9WPw?pwd=3z6m).


## Citations
```
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Fangyi and Zhang, Han and Hu, Kai and Huang, Yu-Kai and Zhu, Chenchen and Savvides, Marios},
    title     = {Enhanced Training of Query-Based Object Detection via Selective Query Recollection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23756-23765}
}
```
