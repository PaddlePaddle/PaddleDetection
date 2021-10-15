# PicoDet

![](../../docs/images/picedet_demo.jpeg)
## Introduction

We developed a series of lightweight models, which named `PicoDet`. Because of its excellent performance, it is very suitable for deployment on mobile or CPU.

Optimizing method of we use:
- [ATSS](https://arxiv.org/abs/1912.02424)
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388)
- Lr Cosine Decay and cycle-EMA
- lightweight head

## Requirements
- PaddlePaddle == 2.1.2
- PaddleSlim >= 2.1.1

## Comming soon
- [ ] Benchmark of PicoDet.
- [ ] deploy for most platforms, such as PaddleLite、MNN、ncnn、openvino etc.
- [ ] PicoDet-XS and PicoDet-L series of model.
- [ ] Slim for PicoDet.
- [ ] More features in need.

## Model Zoo

### PicoDet-S

| Backbone                  | Input size | lr schedule | Box AP(0.5:0.95) | Box AP(0.5) | FLOPS | Model Size | Inference Time |                           download                          | config |
| :------------------------ | :-------: | :-------: | :------: | :---: | :---: | :---: | :------------:  | :-------------------------------------------------: | :-----: |
| ShuffleNetv2-1x    | 320*320    |   280e      |   22.8     | 37.7 | -- | 3.8M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_s_shufflenetv2_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_shufflenetv2_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_shufflenetv2_320_coco.yml) |
| ShuffleNetv2-1x    | 416*416    |   280e      |   25.3     | 41.1 | -- | 3.8M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_s_shufflenetv2_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_shufflenetv2_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_shufflenetv2_416_coco.yml) |


### PicoDet-M

| Backbone                  | Input size | lr schedule | Box AP(0.5:0.95) | Box AP(0.5) | FLOPS | Model Size | Inference Time |                           download                          | config |
| :------------------------ | :-------: | :-------: | :-----------: | :---: | :---: | :---: | :-----: | :-------------------------------------------------: | :-----: |
| ShuffleNetv2-1.5x    | 320*320   |   280e      |   25.3     | 41.2 |  -- | 8.1M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_shufflenetv2_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_shufflenetv2_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_shufflenetv2_320_coco.yml) |
| MobileNetv3-large-1x    | 320*320   |   280e      |   26.7     | 44.1 |  -- | 11.6M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_mbv3_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_mbv3_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_mbv3_320_coco.yml) |
| ShuffleNetv2-1.5x    | 416*416     |   280e      |   28.0     | 44.3 |  -- | 8.1M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_shufflenetv2_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_shufflenetv2_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_shufflenetv2_416_coco.yml) |
| MobileNetv3-large-1x    | 416*416  |   280e      |   29.3     | 47.2 |  -- | 11.6M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_mbv3_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_mbv3_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_mbv3_416_coco.yml) |


**Notes:**

- PicoDet inference speed is tested on Kirin 980 with 4 threads by arm8 and with FP16.
- PicoDet is trained on COCO train2017 dataset and evaluated on val2017.
- PicoDet used 4 GPUs for training and mini-batch size as 128 or 96 on each GPU.

## Citations
```
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}

```
