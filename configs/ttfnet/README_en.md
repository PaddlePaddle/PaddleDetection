# 1. TTFNet

## Introduction

TTFNet is a network used for real-time object detection and friendly to training time. It improves the slow convergence speed of CenterNet and proposes a new method to generate training samples using Gaussian kernel, which effectively eliminates the fuzziness existing in Anchor Free head. At the same time, the simple and lightweight network structure is also easy to expand the task.


**Characteristics:**

The structure is simple, requiring only two heads to detect target position and size, and eliminating time-consuming post-processing operations
The training time is short. Based on DarkNet53 backbone network, V100 8 cards only need 2 hours of training to achieve better model effect

## Model Zoo

| Backbone  | Network type | Number of images per GPU | Learning rate strategy | Inferring time(fps) | Box AP |                                     Download                                     |                                                       Configuration File                                                       |
| :-------- | :----------- | :----------------------: | :--------------------: | :-----------------: | :----: | :------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| DarkNet53 | TTFNet       |            12            |           1x           |        ----         |  33.5  | [link](https://paddledet.bj.bcebos.com/models/ttfnet_darknet53_1x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/ttfnet_darknet53_1x_coco.yml) |





# 2. PAFNet

## Introduction

PAFNet (Paddle Anchor Free) is an optimized model of PaddleDetection based on TTF Net, whose accuracy reaches the SOTA level in the Anchor Free field, and meanwhile produces mobile lightweight model PAFNet-Lite

PAFNet series models optimize TTFNet model from the following aspects:

- [CutMix](https://arxiv.org/abs/1905.04899)
- Better backbone network: ResNet50vd-DCN
- Larger training batch size: 8 GPUs, each GPU batch size=18
- Synchronized Batch Normalization
- [Deformable Convolution](https://arxiv.org/abs/1703.06211)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- Better pretraining model


## Model library

| Backbone   | Net type | Number of images per GPU | Learning rate strategy | Inferring time(fps) | Box AP |                                Download                                 |                                                  Configuration File                                                   |
| :--------- | :------- | :----------------------: | :--------------------: | :-----------------: | :----: | :---------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| ResNet50vd | PAFNet   |            18            |          10x           |        ----         |  39.8  | [link](https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/pafnet_10x_coco.yml) |



### PAFNet-Lite

| Backbone    | Net type    | Number of images per GPU | Learning rate strategy | Box AP | kirin 990 delay（ms） | volume（M） |                                         Download                                          |                                                           Configuration File                                                            |
| :---------- | :---------- | :----------------------: | :--------------------: | :----: | :-------------------: | :---------: | :---------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| MobileNetv3 | PAFNet-Lite |            12            |          20x           |  23.9  |         26.00         |     14      | [link](https://paddledet.bj.bcebos.com/models/pafnet_lite_mobilenet_v3_20x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/pafnet_lite_mobilenet_v3_20x_coco.yml) |

**Attention:** Due to the overall upgrade of the dynamic graph framework, the weighting model published by PaddleDetection of PAF Net needs to be evaluated with a --bias field, for example

```bash
# Published weights using Paddle Detection
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/pafnet_10x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams --bias
```

## Citations
```
@article{liu2019training,
  title   = {Training-Time-Friendly Network for Real-Time Object Detection},
  author  = {Zili Liu, Tu Zheng, Guodong Xu, Zheng Yang, Haifeng Liu, Deng Cai},
  journal = {arXiv preprint arXiv:1909.00700},
  year    = {2019}
}
```
