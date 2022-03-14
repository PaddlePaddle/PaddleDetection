English | [简体中文](README_cn.md)

# ReID of DeepSORT

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) is composed of detector and ReID model in series. Several common ReID models are provided here for the configs of DeepSORT as a reference.

## Model Zoo

### Results on Market1501 pedestrian ReID dataset

| Backbone        | Model                   |   Params  |   FPS     |    mAP    |   Top1    |   Top5    | download  |  config   |
| :-------------: |  :-----------------:    | :-------: |  :------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ResNet-101      |  PCB Pyramid Embedding  |  289M     |   ---     |   86.31   |   94.95   |   98.28   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)   |   [config](./deepsort_pcb_pyramid_r101.yml)     |
| PPLCNet-2.5x    |  PPLCNet Embedding      |  36M      |   ---     |   71.59   |   87.38   |   95.49   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)   |   [config](./deepsort_pplcnet.yml)     |

### Results on VERI-Wild vehicle ReID dataset

| Backbone        | Model                   |  Params   |   FPS     |    mAP    |   Top1    |   Top5    | download  |  config   |
| :-------------: |  :-----------------:    | :-------: |  :------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| PPLCNet-2.5x    |  PPLCNet Embedding      |  93M      |   ---     |   82.44   |   93.54   |   98.53   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.pdparams)   |   [config](./deepsort_pplcnet_vehicle.yml)     |

**Notes:**
  - ReID models are provided by [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), the specific training process and code will be published by PaddleClas.
  - For pedestrian tracking, please use the **Market1501** pedestrian ReID model in combination with a pedestrian detector.
  - For vehicle tracking, please use the **VERI-Wild** vehicle ReID model in combination with a vehicle detector.
