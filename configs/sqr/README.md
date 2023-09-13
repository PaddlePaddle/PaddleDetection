# Enhanced Training of Query-Based Object Detection via Selective Query Recollection


## Introduction
This paper investigates a phenomenon where query-based object detectors mispredict at the last decoding stage while predicting correctly at an intermediate stage. It design and present Selective Query Recollection (SQR), a simple and effective training strategy for query-based object detectors. It cumulatively collects intermediate queries as decoding stages go deeper and selectively forwards the queries to the downstream stages aside from the sequential structure.


## Model Zoo

| Backbone |      Model          | Images/GPU | Epochs | Box AP |            Config                                |   Log   | Download  |
|:--------:|:-------------------:|:----------:|:------:|:------:|:------------------------------------------------:|:-------:|:---------:|
|   R-50   | Deformable DETR SQR |     2      |   50   |  32.9  | [config](./deformable_detr_sqr_r50_12e_coco.yml) | [log]() | [model]() |
|   R-50   | Deformable DETR SQR |     2      |   50   |  45.9  | [config](./deformable_detr_sqr_r50_1x_coco.yml)  | [log]() | [model]() |



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
