# FCOS for Object Detection

## Introduction

FCOS (Fully Convolutional One-Stage Object Detection) is a fast anchor-free object detection framework with strong performance. We reproduced the model of the paper, and improved and optimized the accuracy of the FCOS.

**Highlights:**

- Training Time: The training time of the model of `fcos_r50_fpn_1x` on Tesla v100 with 8 GPU is only 8.5 hours.

## Model Zoo

| Backbone        | Model      | images/GPU | lr schedule |FPS | Box AP |                           download                          | config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | FCOS           |    2    |   1x      |     ----     |  39.6  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/fcos/fcos_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+DCN       |    2    |   1x      |     ----     |  44.3  | [download](https://paddledet.bj.bcebos.com/models/fcos_dcn_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+multiscale_train    |    2    |   2x      |     ----     |  41.8  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/fcos/fcos_r50_fpn_multiscale_2x_coco.yml) |

**Notes:**

- FCOS is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

## Citations
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```
