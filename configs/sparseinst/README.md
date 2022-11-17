# SparseInst for Instance Segmentation

## Introduction

SparseInst is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation. We reproduced this model and provided its Python / C++ deployment.

## Model Zoo

| Model        | Backbone     | Input | with Augmentation | AP<sup>val</sup> | Weight | Config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :------------: | :------------: |
| SparseInst(Base) | ResNet-50 | 640 | False | 32.2 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50_150e_coco.pdparams) | [config](./sparseinst_r50_150e_coco.yml) |
| SparseInst(Base) | ResNet-50 | 640 | True | 32.7 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50_150e_coco_aug.pdparams) | [config](./sparseinst_r50_150e_coco_aug.yml) |
| SparseInst(G-IAM) | ResNet-50 | 640 | False | 32.7 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50_giam_150e_coco.pdparams) | [config](./sparseinst_r50_giam_150e_coco.yml) |
| SparseInst(G-IAM) | ResNet-50 | 640 | True | 33.3 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50_giam_150e_coco_aug.pdparams) | [config](./sparseinst_r50_giam_150e_coco_aug.yml) |
| SparseInst(G-IAM) | ResNet-50-vd | 640 | True | 34.3 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50vd_giam_150e_coco_aug.pdparams) | [config](./sparseinst_r50vd_giam_150e_coco_aug.yml) |
| SparseInst(G-IAM) | ResNet-50-vd-ssld | 640 | True | 36.4 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50vd_ssld_giam_150_coco_aug.pdparams) | [config](./sparseinst_r50vd_ssld_giam_150_coco_aug.yml) |
| SparseInst(G-IAM) | ResNet-50-vd-ssld-dcn | 640 | True | 37.9 | [Download](https://bj.bcebos.com/v1/paddledet/models/sparseinst_r50vd_ssld_dcn_giam_150e_coco_aug.pdparams) | [config](./sparseinst_r50vd_ssld_dcn_giam_150e_coco_aug.yml) |


**Notes:**

- SparseInst is trained on COCO train2017 dataset on **8 GPU** and evaluated on val2017.
- For inference on TensorRT, TensorRT 7+ and PaddlePaddle >=2.4 are required.

## Citations
```
@inproceedings{Cheng2022SparseInst,
  title     =   {Sparse Instance Activation for Real-Time Instance Segmentation},
  author    =   {Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu},
  booktitle =   {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2022}
}
```
