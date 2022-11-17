# SparseInst for Instance Segmentation

## Introduction

SparseInst is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation. We reproduced this model and provided its Python / C++ deployment.

**Highlights:**

- Training Time: The training time of the model of `fcos_r50_fpn_1x` on Tesla v100 with 8 GPU is only 8.5 hours.

## Model Zoo

| Model        | Backbone     | Input | with Augmentation | AP<sup>val</sup> | Weight | Config |
| :-------------- | :------------- | :-----: | :-----: | :------------: |
| SparseInst(Base) | ResNet-50 | 640 | False | 32.2 | [Download]() | [config]() |
| SparseInst(Base) | ResNet-50 | 640 | True | 32.7 | [Download]() | [config]() |
| SparseInst(G-IAM) | ResNet-50 | 640 | False | 32.7 | [Download]() | [config]() |
| SparseInst(G-IAM) | ResNet-50 | 640 | True | 33.3 | [Download]() | [config]() |
| SparseInst(G-IAM) | ResNet-50-vd | 640 | True | 34.3 | [Download]() | [config]() |
| SparseInst(G-IAM) | ResNet-50-vd-ssld | 640 | True | 36.4 | [Download]() | [config]() |
| SparseInst(G-IAM) | ResNet-50-vd-ssld-dcn | 640 | True | 37.9 | [Download]() | [config]() |


**Notes:**

- SparseInst is trained on COCO train2017 dataset on **8 GPU** and evaluated on val2017.


## Citations
```
@inproceedings{Cheng2022SparseInst,
  title     =   {Sparse Instance Activation for Real-Time Instance Segmentation},
  author    =   {Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu},
  booktitle =   {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2022}
}
```
