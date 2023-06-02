[简体中文](PaddleYOLO_MODEL.md) | English

# [**PaddleYOLO**](https://github.com/PaddlePaddle/PaddleYOLO)

## Introduction
- [**PaddleYOLO**](#paddleyolo)
  - [Introduction](#introduction)
  - [Introduction](#introduction-1)
  - [Updates](#updates)
  - [ModelZoo](#modelzoo)
    - [PP-YOLOE](#pp-yoloe)
    - [YOLOX](#yolox)
    - [YOLOv5](#yolov5)
    - [YOLOv6](#yolov6)
    - [YOLOv7](#yolov7)
    - [YOLOv8](#yolov8)
    - [RTMDet](#rtmdet)
    - [**Notes：**](#notes)
    - [VOC](#voc)
  - [UserGuide](#userguide)
    - [**Pipeline**](#pipeline)
    - [CustomDataset](#customdataset)
      - [preparation：](#preparation)
      - [fintune：](#fintune)
      - [Predict and export:](#predict-and-export)

## Introduction

**PaddleYOLO** is a YOLO Series toolbox based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **only relevant codes of YOLO series models are included**. It supports `YOLOv3`,`PP-YOLO`,`PP-YOLOv2`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`,`YOLOv8`,`RTMDet` and so on. Welcome to use and build it together!

## Updates

* 【2023/01/10】Support [YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) inference and deploy;
* 【2022/09/29】Support [RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet) inference and deploy;
* 【2022/09/26】Release [`PaddleYOLO`](https://github.com/PaddlePaddle/PaddleYOLO);
* 【2022/09/19】Support the new version of [`YOLOv6`](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6), including n/t/s/m/l model;
* 【2022/08/23】Release `YOLOSeries` codebase: support `YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6` and `YOLOv7`; support using `ConvNeXt` backbone to get high-precision version of `PP-YOLOE`,`YOLOX` and `YOLOv5`; support PaddleSlim accelerated quantitative training `PP-YOLOE`,`YOLOv5`,`YOLOv6` and `YOLOv7`. For details, please read this [article](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


**Notes：**
 - The Licence of **PaddleYOLO** is **GPL 3.0**, the codes of [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5),[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7) and [YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8) will not be merged into [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). Except for these three YOLO models, other YOLO models are recommended to use in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **which will be the first to release the latest progress of PP-YOLO series detection model**;
 - To use **PaddleYOLO**, **PaddlePaddle-2.3.2 or above is recommended**，please refer to the [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to download the appropriate version. **For Windows platforms, please install the paddle develop version**;
 - Training **Custom dataset** please refer to [doc](#CustomDataset) and [issue](https://github.com/PaddlePaddle/PaddleYOLO/issues/43). Please **ensure COCO trained weights are loaded as pre-train** at first. We recommend to use YOLO detection model **with a total `batch_size` at least greater than `64` to train**. If the resources are insufficient, please **use the smaller model** or **reduce the input size of the model**. To ensure high detection accuracy, **you'd better never try to using single GPU or total `batch_size` less than `32` for training**;

## ModelZoo

### [PP-YOLOE](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| PP-YOLOE-s   |     640   |    32    |  400e    |    2.9    |       43.4        |        60.0         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml)                   |
| PP-YOLOE-s   |     640   |    32    |  300e    |    2.9    |       43.0        |        59.6         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m   |      640  |    28    |  300e    |    6.0    |       49.0        |        65.9         |   23.43   |  49.91   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l   |      640  |    20    |  300e    |    8.7    |       51.4        |        68.6         |   52.20   |  110.07 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x   |      640  |    16    |  300e    |    14.9   |       52.3        |        69.5         |   98.42   |  206.59  |[model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)    |
| PP-YOLOE-tiny ConvNeXt| 640 |    16      |   36e    | -   |       44.6        |        63.3         |   33.04   |  13.87 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_convnext_tiny_36e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/convnext/ppyoloe_convnext_tiny_36e_coco.yml) |
| **PP-YOLOE+_s**   |     640   |    8    |  80e    |    2.9    |     **43.7**    |      **60.6**     |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)                   |
| **PP-YOLOE+_m**   |      640  |    8    |  80e    |    6.0    |     **49.8**    |      **67.1**     |   23.43   |  49.91   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml)                   |
| **PP-YOLOE+_l**   |      640  |    8    |  80e    |    8.7    |     **52.9**    |      **70.1**     |   52.20   |  110.07 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml)                   |
| **PP-YOLOE+_x**   |      640  |    8    |  80e    |    14.9   |     **54.7**    |      **72.0**     |   98.42   |  206.59  |[model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml)                   |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| PP-YOLOE-s(400epoch) |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_400e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_400e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_400e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_400e_coco_wo_nms.onnx) |
| PP-YOLOE-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_s_300e_coco_wo_nms.onnx) |
| PP-YOLOE-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_m_300e_coco_wo_nms.onnx) |
| PP-YOLOE-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_l_300e_coco_wo_nms.onnx) |
| PP-YOLOE-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_crn_x_300e_coco_wo_nms.onnx) |
| **PP-YOLOE+_s** |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_s_80e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_s_80e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_s_80e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_s_80e_coco_wo_nms.onnx) |
| **PP-YOLOE+_m** |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_m_80e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_m_80e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_m_80e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_m_80e_coco_wo_nms.onnx) |
| **PP-YOLOE+_l** |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_l_80e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_l_80e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_l_80e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_l_80e_coco_wo_nms.onnx) |
| **PP-YOLOE+_x** |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_x_80e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_x_80e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_x_80e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/ppyoloe/ppyoloe_plus_crn_x_80e_coco_wo_nms.onnx) |

</details>

### [YOLOX](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOX-nano     |  416     |    8      |   300e    |     2.3    |  26.1  |  42.0 |  0.91  |  1.08 | [model](https://paddledet.bj.bcebos.com/models/yolox_nano_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_nano_300e_coco.yml) |
| YOLOX-tiny     |  416     |    8      |   300e    |     2.8    |  32.9  |  50.4 |  5.06  |  6.45 | [model](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_tiny_300e_coco.yml) |
| YOLOX-s        |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  9.0  |  26.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_s_300e_coco.yml) |
| YOLOX-m        |  640     |    8      |   300e    |     5.8    |  46.9  |  65.7 |  25.3  |  73.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_m_300e_coco.yml) |
| YOLOX-l        |  640     |    8      |   300e    |     9.3    |  50.1  |  68.8 |  54.2  |  155.6 | [model](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_l_300e_coco.yml) |
| YOLOX-x        |  640     |    8      |   300e    |     16.6   |  **51.8**  |  **70.6** |  99.1  |  281.9 | [model](https://paddledet.bj.bcebos.com/models/yolox_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_x_300e_coco.yml) |
 YOLOX-cdn-tiny    |  416     |    8      |   300e    |     1.9    |  32.4  |  50.2 |  5.03 |  6.33  | [model](https://paddledet.bj.bcebos.com/models/yolox_cdn_tiny_300e_coco.pdparams) | [config](c../../onfigs/yolox/yolox_cdn_tiny_300e_coco.yml) |
| YOLOX-crn-s     |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  7.7  |  24.69 | [model](https://paddledet.bj.bcebos.com/models/yolox_crn_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox/yolox_crn_s_300e_coco.yml) |
| YOLOX-s ConvNeXt|  640     |    8      |   36e     |     -      |  44.6  |  65.3 |  36.2 |  27.52 | [model](https://paddledet.bj.bcebos.com/models/yolox_convnext_s_36e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/convnext/yolox_convnext_s_36e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOx-nano |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_wo_nms.onnx) |
| YOLOx-tiny |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_wo_nms.onnx) |
| YOLOx-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_wo_nms.onnx) |
| YOLOx-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_wo_nms.onnx) |
| YOLOx-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_wo_nms.onnx) |
| YOLOx-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_wo_nms.onnx) |

</details>


### [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-n        |  640     |    16     |   300e    |     2.6    |  28.0  | 45.7 |  1.87  | 4.52 | [model](https://paddledet.bj.bcebos.com/models/yolov5_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    16      |   300e    |     3.2    |  37.6  | 56.7 |  7.24  | 16.54 | [model](https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    16      |   300e    |     5.2    |  45.4  | 64.1 |  21.19  | 49.08 | [model](https://paddledet.bj.bcebos.com/models/yolov5_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    16      |   300e    |     7.9    |  48.9  | 67.1 |  46.56  | 109.32 | [model](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    16      |   300e    |     13.7   |  50.6  | 68.7 |  86.75  | 205.92 | [model](https://paddledet.bj.bcebos.com/models/yolov5_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_x_300e_coco.yml) |
| YOLOv5-s ConvNeXt|  640    |    8      |   36e     |     -      |  42.4  |  65.3  |  34.54 |  17.96 | [model](https://paddledet.bj.bcebos.com/models/yolov5_convnext_s_36e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5_convnext_s_36e_coco.yml) |
| *YOLOv5p6-n        |  1280     |    16     |   300e    |     -    |  35.9  | 54.2 |  3.25  | 9.23 | [model](https://paddledet.bj.bcebos.com/models/yolov5p6_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5p6_n_300e_coco.yml) |
| *YOLOv5p6-s        |  1280     |    16     |   300e    |     -    |  44.5  | 63.3 |  12.63  | 33.81 | [model](https://paddledet.bj.bcebos.com/models/yolov5p6_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5p6_s_300e_coco.yml) |
| *YOLOv5p6-m        |  1280     |    16     |   300e    |     -    |  51.1  | 69.0 |  35.73  | 100.21 | [model](https://paddledet.bj.bcebos.com/models/yolov5p6_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5p6_m_300e_coco.yml) |
| *YOLOv5p6-l        |  1280     |    8      |   300e    |     -    |  53.4  | 71.0 |  76.77  | 223.09 | [model](https://paddledet.bj.bcebos.com/models/yolov5p6_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5p6_l_300e_coco.yml) |
| *YOLOv5p6-x        |  1280     |    8      |   300e    |     -    |  54.7  | 72.4 |  140.80 | 420.03 | [model](https://paddledet.bj.bcebos.com/models/yolov5p6_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5/yolov5p6_x_300e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv5-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.onnx) |
| YOLOv5-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.onnx) |
| YOLOv5-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.onnx) |
| YOLOv5-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.onnx) |
| YOLOv5-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.onnx) |

</details>

### [YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *YOLOv6-n       |  640     |    16      |   300e(+300e) |  2.0  |  37.5 |    53.1 |  5.07  | 12.49 |[model](https://paddledet.bj.bcebos.com/models/yolov6_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_n_300e_coco.yml) |
| *YOLOv6-s       |  640     |    32      |   300e(+300e) |  2.7  |  44.8 |    61.7 |  20.18  | 49.36 |[model](https://paddledet.bj.bcebos.com/models/yolov6_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_s_300e_coco.yml) |
| *YOLOv6-m       |  640     |    32      |   300e(+300e) |  -  |  49.5 |    66.9 |  37.74  | 92.47 |[model](https://paddledet.bj.bcebos.com/models/yolov6_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_m_300e_coco.yml) |
| *YOLOv6-l(silu) |  640     |    32      |   300e(+300e) |  -  |  52.2 |    70.2 |  59.66  | 149.4 |[model](https://paddledet.bj.bcebos.com/models/yolov6_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_l_300e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| yolov6-n |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_wo_nms.onnx) |
| yolov6-s |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_wo_nms.onnx) |
| yolov6-m |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.onnx) |
| yolov6-l(silu) |  640  | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.onnx) |

</details>

### [YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv7-L        |  640     |    32      |   300e    |     7.4     |  51.0  | 70.2 |  37.62  | 106.08 |[model](https://paddledet.bj.bcebos.com/models/yolov7_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7_l_300e_coco.yml) |
| *YOLOv7-X        |  640     |    32      |   300e    |     12.2    |  53.0  | 70.8 |  71.34  | 190.08 | [model](https://paddledet.bj.bcebos.com/models/yolov7_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7_x_300e_coco.yml) |
| *YOLOv7P6-W6     |  1280    |    16      |   300e    |     25.5    |  54.4  | 71.8 |  70.43  | 360.26 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_w6_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7p6_w6_300e_coco.yml) |
| *YOLOv7P6-E6     |  1280    |    10      |   300e    |     31.1    |  55.7  | 73.0 |  97.25  | 515.4 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7p6_e6_300e_coco.yml) |
| *YOLOv7P6-D6     |  1280    |    8      |   300e    |     37.4    | 56.1  | 73.3 |  133.81  | 702.92 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_d6_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7p6_d6_300e_coco.yml) |
| *YOLOv7P6-E6E    |  1280    |    6      |   300e    |     48.7    |  56.5  | 73.7 |  151.76  | 843.52 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6e_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7p6_e6e_300e_coco.yml) |
| YOLOv7-tiny     |  640     |    32      |   300e    |     -   |  37.3 | 54.5 |  6.23  | 6.90 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7_tiny_300e_coco.yml) |
| YOLOv7-tiny     |  416     |    32      |   300e    |     -    | 33.3 | 49.5 |  6.23  | 2.91 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_416_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7_tiny_416_300e_coco.yml) |
| YOLOv7-tiny     |  320     |    32      |   300e    |     -    | 29.1 | 43.8 |  6.23  | 1.73 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_320_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7/yolov7_tiny_320_300e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv7-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_l_300e_coco_wo_nms.onnx) |
| YOLOv7-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_x_300e_coco_wo_nms.onnx) |
| YOLOv7P6-W6 |  1280   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_w6_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_w6_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_w6_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_w6_300e_coco_wo_nms.onnx) |
| YOLOv7P6-E6 |  1280   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6_300e_coco_wo_nms.onnx) |
| YOLOv7P6-D6 |  1280   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_d6_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_d6_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_d6_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_d6_300e_coco_wo_nms.onnx) |
| YOLOv7P6-E6E |  1280   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6e_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6e_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6e_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7p6_e6e_300e_coco_wo_nms.onnx) |
| YOLOv7-tiny |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_300e_coco_wo_nms.onnx) |
| YOLOv7-tiny |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_416_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_416_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_416_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_416_300e_coco_wo_nms.onnx) |
| YOLOv7-tiny |  320   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_320_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_320_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_320_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov7/yolov7_tiny_320_300e_coco_wo_nms.onnx) |

</details>


### [YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-n        |  640     |    16      |   500e   |    2.4   |  37.3  | 53.0 |  3.16   | 8.7 | [model](https://paddledet.bj.bcebos.com/models/yolov8_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_n_300e_coco.yml) |
| *YOLOv8-s        |  640     |    16      |   500e   |    3.4   |  44.9  | 61.8 |  11.17  | 28.6 | [model](https://paddledet.bj.bcebos.com/models/yolov8_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_s_300e_coco.yml) |
| *YOLOv8-m        |  640     |    16      |   500e   |    6.5   |  50.2  | 67.3 |  25.90  | 78.9 | [model](https://paddledet.bj.bcebos.com/models/yolov8_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_m_300e_coco.yml) |
| *YOLOv8-l        |  640     |    16      |   500e   |    10.0  |  52.8  | 69.6 |  43.69  | 165.2 | [model](https://paddledet.bj.bcebos.com/models/yolov8_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_l_300e_coco.yml) |
| *YOLOv8-x        |  640     |    16      |   500e   |    15.1  |  53.8  | 70.6 |  68.23  | 257.8 | [model](https://paddledet.bj.bcebos.com/models/yolov8_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_x_300e_coco.yml) |
| *YOLOv8-P6-x     |  1280    |    16      |   500e   |    55.0  |    -   |   -  |  97.42  | 522.93 | [model](https://paddledet.bj.bcebos.com/models/yolov8p6_x_500e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8p6_x_500e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv8-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_wo_nms.onnx) |
| YOLOv8-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_wo_nms.onnx) |
| YOLOv8-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_wo_nms.onnx) |
| YOLOv8-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_wo_nms.onnx) |
| YOLOv8-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_wo_nms.onnx) |

</details>


### [RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    download       | config |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *RTMDet-t       |  640     |    32      |   300e    |    2.8   |  40.9 | 57.9 |  4.90  | 16.21 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_t_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_t_300e_coco.yml) |
| *RTMDet-s       |  640     |    32      |   300e    |    3.3   |  44.5 | 62.0 |  8.89  | 29.71 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_s_300e_coco.yml) |
| *RTMDet-m       |  640     |    32      |   300e    |    6.4   |  49.1 | 66.8 |  24.71  | 78.47 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_m_300e_coco.yml) |
| *RTMDet-l       |  640     |    32      |   300e    |    10.2  |  51.2 | 68.8 |  52.31  | 160.32 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_l_300e_coco.yml) |
| *RTMDet-x       |  640     |    32      |   300e    |    18.0  |  52.6 | 70.4 |  94.86  | 283.12 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_x_300e_coco.yml) |

</details>

<details>
<summary> Deploy Models  </summary>

| Model     | Input Size | Exported weights(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| RTMDet-t |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.onnx) |
| RTMDet-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.onnx) |
| RTMDet-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.onnx) |
| RTMDet-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.onnx) |
| RTMDet-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.onnx) |

</details>


### **Notes：**
 - All the models are trained on COCO train2017 dataset and evaluated on val2017 dataset. The * in front of the model indicates that the training is being updated.
 - Please check the specific accuracy and speed details in [PP-YOLOE](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe),[YOLOX](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox),[YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5),[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7). **Note that YOLOv5, YOLOv6 and YOLOv7 have not adopted `multi_label` to eval**.
- TRT-FP16-Latency(ms) is the time spent in testing under TensorRT-FP16, **excluding data preprocessing and model output post-processing (NMS)**. The test adopts single card **Tesla T4 GPU, batch size=1**, and the test environment is **paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**. Please refer to the respective model homepage for details.
- For **FLOPs(G) and Params(M)**, you should first install [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), `pip install paddleslim`, then set `print_flops: True` and `print_params: True` in [runtime.yml](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/runtime.yml). Make sure **single scale** like 640x640, **MACs are printed，FLOPs=2*MACs**.
 - Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), quantitative training of YOLO series models can achieve basically lossless accuracy and generally improve the speed by more than 30%. For details, please refer to [auto_compression](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression).


### [VOC](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc)

<details>
<summary> Baseline </summary>

| Model        | Input Size  | images/GPU | Epoch | TRT-FP16-Latency(ms) | mAP(0.50,11point) | Params(M) | FLOPs(G) |    download       | config |
| :-----------: | :-------: | :-------: | :------: | :------------: | :---------------: | :------------------: |:-----------------: | :------: | :------: |
| YOLOv5-s        |  640     |    16     |   60e    |     3.2   |  80.3 |  7.24  | 16.54 | [model](https://paddledet.bj.bcebos.com/models/yolov5_s_60e_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolov5_s_60e_voc.yml) |
| YOLOv7-tiny     |  640     |    32     |   60e    |     2.6   |  80.2 |  6.23  | 6.90 | [model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_60e_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolov7_tiny_60e_voc.yml) |
| YOLOX-s         |  640     |    8      |   40e    |     3.0   |  82.9 |  9.0   |  26.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_s_40e_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolox_s_40e_voc.yml) |
| PP-YOLOE+_s     |  640     |    8      |   30e    |     2.9   |  86.7 |  7.93  |  17.36 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_30e_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/ppyoloe_plus_crn_s_30e_voc.yml) |

</details>

**Note:**
  - The VOC mAP is `mAP(IoU=0.5)`, and all the models **have not adopted `multi_label` to eval**.
  - All YOLO VOC models are loaded with the COCO weights of their respective models as pre-train weights. Each config file uses 8 GPUs by default, which can be used as a reference for setting custom datasets. The specific mAP will vary depending on the datasets;
  - We recommend to use YOLO detection model **with a total `batch_size` at least greater than `64` to train**. If the resources are insufficient, please **use the smaller model** or **reduce the input size of the model**. To ensure high detection accuracy, **you'd better not try to using single GPU or total `batch_size` less than `64` for training**;
  - Params (M) and FLOPs (G) are measured during training. YOLOv7 has no s model, so tiny model is selected;
  - For TRT-FP16 Latency (ms) speed measurement, please refer to the config homepage of each YOLO model;


## UserGuide

Download MS-COCO dataset, [official website](https://cocodataset.org). The download links are: [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip), [test2017](http://images.cocodataset.org/zips/test2017.zip).
The download link provided by PaddleDetection team is: [coco](https://bj.bcebos.com/v1/paddledet/data/coco.tar)(about 22G) and [test2017](https://bj.bcebos.com/v1/paddledet/data/cocotest2017.zip). Note that test2017 is optional, and the evaluation is based on val2017.


### **Pipeline**

Write the following commands in a script file, such as ```run.sh```, and run as：```sh run.sh```. You can also run the command line sentence by sentence.

```bash
model_name=ppyoloe # yolov7
job_name=ppyoloe_plus_crn_l_80e_coco # yolov7_tiny_300e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.training（single GPU / multi GPU）
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.eval
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.infer
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5

# 4.export
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# 5.deploy infer
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.deploy speed, add '--run_mode=trt_fp16' to test in TensorRT FP16 mode
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.export onnx
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx speed
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
```

**Note：**
- If you want to switch models, just modify the first two lines, such as:
  ```
  model_name=yolov7
  job_name=yolov7_tiny_300e_coco
  ```
- For **exporting onnx**, you should install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) by `pip install paddle2onnx` at first.
- For **FLOPs(G) and Params(M)**, you should install [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) by `pip install paddleslim` at first, then set `print_flops: True` and `print_params: True` in [runtime.yml](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/runtime.yml). Make sure **single scale** like 640x640, **MACs are printed，FLOPs=2*MACs**.


### CustomDataset

#### preparation：

1.For the annotation of custom dataset, please refer to[DetAnnoTools](../tutorials/data/DetAnnoTools.md);

2.For training preparation of custom dataset，please refer to[PrepareDataSet](../tutorials/PrepareDataSet.md).


#### fintune：

In addition to changing the path of the dataset, it is generally recommended to load **the COCO pre training weight of the corresponding model** to fintune, which will converge faster and achieve higher accuracy, such as：

```base
# fintune with single GPU：
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# fintune with multi GPU：
python -m paddle.distributed.launch --log_dir=./log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

**Note:**
- The fintune training will show that the channels of the last layer of the head classification branch is not matched, which is a normal situation, because the number of custom dataset is generally inconsistent with that of COCO dataset;
- In general, the number of epochs for fintune training can be set less, and the lr setting is also smaller, such as 1/10. The highest accuracy may occur in one of the middle epochs;

#### Predict and export:

When using custom dataset to predict and export models, if the path of the TestDataset dataset is set incorrectly, COCO 80 categories will be used by default.

In addition to the correct path setting of the TestDataset dataset, you can also modify and add the corresponding `label_list`. Txt file (one category is recorded in one line), and `anno_path` in TestDataset can also be set as an absolute path, such as:
```
TestDataset:
  !ImageFolder
    anno_path: label_list.txt # if not set dataset_dir, the anno_path will be relative path of PaddleDetection root directory
    # dataset_dir: dataset/my_coco # if set dataset_dir, the anno_path will be dataset_dir/anno_path
```
one line in `label_list.txt` records a corresponding category：
```
person
vehicle
```
