# FCOS for Object Detection

## Introduction

FCOS (Fully Convolutional One-Stage Object Detection) is a fast anchor-free object detection framework with strong performance. We reproduced the model of the paper, and improved and optimized the accuracy of the FCOS.

**Highlights:**

- Training Time: The training time of the model of `fcos_r50_fpn_1x` on Tesla v100 with 8 GPU is only 8.5 hours.

## Model Zoo

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | FCOS           |    2    |   1x      |     ----     |  39.6  | [下载链接](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/fcos/fcos_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+DCN       |    2    |   1x      |     ----     |  44.3  | [下载链接](https://paddledet.bj.bcebos.com/models/fcos_dcn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+multiscale_train    |    2    |   2x      |     ----     |  41.8  | [下载链接](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/fcos/fcos_r50_fpn_multiscale_2x_coco.yml) |

**Notes:**

- FCOS is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- FCOS training performace is dependented on Paddle develop branch, performance reproduction shoule based on [Paddle daily version](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev) or Paddle 2.0.1(will be published on 2021.03), performace may loss slightly if training base on Paddle 2.0.0

## Citations
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```
