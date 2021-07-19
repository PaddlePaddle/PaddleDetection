# PicoDet

## Introduction

We developed a series of mobile models, which named `PicoDet`.
Optimizing method of we use:
- [Generalized Focal Loss V2](https://arxiv.org/pdf/2011.12885.pdf)
- Lr Cosine Decay



## Model Zoo

### PicoDet-S

| Backbone                  | Input size | images/GPU | lr schedule |Box AP | FLOPS | Inference Time |                           download                          | config |
| :------------------------ | :-------: | :-------: | :-----------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-----: |
| ShuffleNetv2-1x    | 320*320   |    128    |   280e      |   21.9     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_s_shufflenetv2_320_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_shufflenetv2_320_coco.yml) |
| MobileNetv3-large-0.5x    | 320*320   |    128    |   280e      |   20.4     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_s_mbv3_320_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_mbv3_320_coco.yml) |
| ShuffleNetv2-1x    | 416*416   |    96    |   280e      |   24.0     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_s_shufflenetv2_416_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_shufflenetv2_416_coco.yml) |
| MobileNetv3-large-0.5x    | 416*416   |    96    |   280e      |   23.3     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_s_mbv3_416_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_mbv3_416_coco.yml) |

### PicoDet-M

| Backbone                  | Input size | images/GPU | lr schedule |Box AP | FLOPS | Inference Time |                           download                          | config |
| :------------------------ | :-------: | :-------: | :-----------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-----: |
| ShuffleNetv2-1.5x    | 320*320   |    128    |   280e      |   24.9     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_m_shufflenetv2_320_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_shufflenetv2_320_coco.yml) |
| MobileNetv3-large-1x    | 320*320   |    128    |   280e      |   26.4     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_m_mbv3_320_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_mbv3_320_coco.yml) |
| ShuffleNetv2-1.5x    | 416*416   |    128    |   280e      |   27.4     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_m_shufflenetv2_416_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_shufflenetv2_416_coco.yml) |
| MobileNetv3-large-1x    | 416*416   |    128    |   280e      |   29.2     |  -- | -- | [download](https://paddledet.bj.bcebos.com/models/picodet_m_mbv3_416_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_mbv3_416_coco.yml) |


**Notes:**

- PicoDet inference speed is tested on Kirin 980 with 4 threads by arm8 and with FP16.
- PicoDet is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- PicoDet used 4 GPUs for training and mini-batch size as 128 or 96 on each GPU.

## Citations
```
@article{li2020gflv2,
  title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2011.12885},
  year={2020}
}

```
