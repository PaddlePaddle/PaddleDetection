简体中文 | [English](PaddleYOLO_MODEL_en.md)

# [**PaddleYOLO**](https://github.com/PaddlePaddle/PaddleYOLO)

## 内容
- [**PaddleYOLO**](#paddleyolo)
  - [内容](#内容)
  - [简介](#简介)
  - [更新日志](#更新日志)
  - [模型库](#模型库)
    - [PP-YOLOE](#pp-yoloe)
    - [YOLOX](#yolox)
    - [YOLOv5](#yolov5)
    - [YOLOv6](#yolov6)
    - [YOLOv7](#yolov7)
    - [YOLOv8](#yolov8)
    - [RTMDet](#rtmdet)
    - [**注意:**](#注意)
    - [VOC](#voc)
  - [使用指南](#使用指南)
    - [**一键运行全流程**](#一键运行全流程)
    - [自定义数据集](#自定义数据集)
      - [数据集准备：](#数据集准备)
      - [fintune训练：](#fintune训练)
      - [预测和导出：](#预测和导出)

## 简介

**PaddleYOLO**是基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的YOLO系列模型库，**只包含YOLO系列模型的相关代码**，支持`YOLOv3`,`PP-YOLO`,`PP-YOLOv2`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`,`YOLOv8`,`RTMDet`等模型，欢迎一起使用和建设！

## 更新日志
* 【2023/01/10】支持[YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8)预测和部署；
* 【2022/09/29】支持[RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet)预测和部署；
* 【2022/09/26】发布[`PaddleYOLO`](https://github.com/PaddlePaddle/PaddleYOLO)模型套件；
* 【2022/09/19】支持[`YOLOv6`](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6)新版，包括n/t/s/m/l模型；
* 【2022/08/23】发布`YOLOSeries`代码库: 支持`YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`等YOLO模型，支持`ConvNeXt`骨干网络高精度版`PP-YOLOE`,`YOLOX`和`YOLOv5`等模型，支持PaddleSlim无损加速量化训练`PP-YOLOE`,`YOLOv5`,`YOLOv6`和`YOLOv7`等模型，详情可阅读[此文章](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


**注意:**
 - **PaddleYOLO**代码库协议为**GPL 3.0**，[YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5),[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7)和[YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8)这几类模型代码不合入[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，其余YOLO模型推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中使用，**会最先发布PP-YOLO系列特色检测模型的最新进展**；；
 - **PaddleYOLO**代码库**推荐使用paddlepaddle-2.3.2以上的版本**，请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载对应适合版本，**Windows平台请安装paddle develop版本**；
 - PaddleYOLO 的[Roadmap](https://github.com/PaddlePaddle/PaddleYOLO/issues/44) issue用于收集用户的需求，欢迎提出您的建议和需求。
 - 训练**自定义数据集**请参照[文档](#自定义数据集)和[issue](https://github.com/PaddlePaddle/PaddleYOLO/issues/43)。请首先**确保加载了COCO权重作为预训练**，YOLO检测模型建议**总`batch_size`至少大于`64`**去训练，如果资源不够请**换小模型**或**减小模型的输入尺度**，为了保障较高检测精度，**尽量不要尝试单卡训和总`batch_size`小于`32`训**；


## 模型库

### [PP-YOLOE](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe)

<details>
<summary> 基础模型 </summary>

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
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
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
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
<summary> 基础模型 </summary>

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
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
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
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
<summary> 基础模型 </summary>

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
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
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv5-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.onnx) |
| YOLOv5-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.onnx) |
| YOLOv5-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.onnx) |
| YOLOv5-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.onnx) |
| YOLOv5-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.onnx) |

</details>

### [YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6)

<details>
<summary> 基础模型 </summary>

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *YOLOv6-n       |  640     |    16      |   300e(+300e) |  2.0  |  37.5 |    53.1 |  5.07  | 12.49 |[model](https://paddledet.bj.bcebos.com/models/yolov6_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_n_300e_coco.yml) |
| *YOLOv6-s       |  640     |    32      |   300e(+300e) |  2.7  |  44.8 |    61.7 |  20.18  | 49.36 |[model](https://paddledet.bj.bcebos.com/models/yolov6_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_s_300e_coco.yml) |
| *YOLOv6-m       |  640     |    32      |   300e(+300e) |  -  |  49.5 |    66.9 |  37.74  | 92.47 |[model](https://paddledet.bj.bcebos.com/models/yolov6_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_m_300e_coco.yml) |
| *YOLOv6-l(silu) |  640     |    32      |   300e(+300e) |  -  |  52.2 |    70.2 |  59.66  | 149.4 |[model](https://paddledet.bj.bcebos.com/models/yolov6_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6/yolov6_l_300e_coco.yml) |

</details>

<details>
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| yolov6-n |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_300e_coco_wo_nms.onnx) |
| yolov6-s |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_300e_coco_wo_nms.onnx) |
| yolov6-m |  640   | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.onnx) |
| yolov6-l(silu) |  640  | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.zip) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.zip) | [(w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.onnx) &#124; [(w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.onnx) |

</details>

### [YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7)

<details>
<summary> 基础模型 </summary>

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
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
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
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
<summary> 基础模型 </summary>

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-n        |  640     |    16      |   500e   |    2.4   |  37.3  | 53.0 |  3.16   | 8.7 | [model](https://paddledet.bj.bcebos.com/models/yolov8_n_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_n_300e_coco.yml) |
| *YOLOv8-s        |  640     |    16      |   500e   |    3.4   |  44.9  | 61.8 |  11.17  | 28.6 | [model](https://paddledet.bj.bcebos.com/models/yolov8_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_s_300e_coco.yml) |
| *YOLOv8-m        |  640     |    16      |   500e   |    6.5   |  50.2  | 67.3 |  25.90  | 78.9 | [model](https://paddledet.bj.bcebos.com/models/yolov8_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_m_300e_coco.yml) |
| *YOLOv8-l        |  640     |    16      |   500e   |    10.0   |  52.8  | 69.6 |  43.69  | 165.2 | [model](https://paddledet.bj.bcebos.com/models/yolov8_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_l_300e_coco.yml) |
| *YOLOv8-x        |  640     |    16      |   500e   |    15.1  |  53.8  | 70.6 |  68.23  | 257.8 | [model](https://paddledet.bj.bcebos.com/models/yolov8_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8_x_300e_coco.yml) |
| *YOLOv8-P6-x     |  1280    |    16      |   500e   |    55.0  |    -   |   -  |  97.42  | 522.93 | [model](https://paddledet.bj.bcebos.com/models/yolov8p6_x_500e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8/yolov8p6_x_500e_coco.yml) |

</details>

<details>
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv8-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_n_500e_coco_wo_nms.onnx) |
| YOLOv8-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_s_500e_coco_wo_nms.onnx) |
| YOLOv8-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_m_500e_coco_wo_nms.onnx) |
| YOLOv8-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_l_500e_coco_wo_nms.onnx) |
| YOLOv8-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov8/yolov8_x_500e_coco_wo_nms.onnx) |

</details>


### [RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet)

<details>
<summary> 基础模型 </summary>

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *RTMDet-t       |  640     |    32      |   300e    |    2.8   |  40.9 | 57.9 |  4.90  | 16.21 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_t_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_t_300e_coco.yml) |
| *RTMDet-s       |  640     |    32      |   300e    |    3.3   |  44.5 | 62.0 |  8.89  | 29.71 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_s_300e_coco.yml) |
| *RTMDet-m       |  640     |    32      |   300e    |    6.4   |  49.1 | 66.8 |  24.71  | 78.47 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_m_300e_coco.yml) |
| *RTMDet-l       |  640     |    32      |   300e    |    10.2  |  51.2 | 68.8 |  52.31  | 160.32 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_l_300e_coco.yml) |
| *RTMDet-x       |  640     |    32      |   300e    |    18.0  |  52.6 | 70.4 |  94.86  | 283.12 |[model](https://paddledet.bj.bcebos.com/models/rtmdet_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet/rtmdet_x_300e_coco.yml) |

</details>

<details>
<summary> 部署模型  </summary>

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| RTMDet-t |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.onnx) |
| RTMDet-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.onnx) |
| RTMDet-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.onnx) |
| RTMDet-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.onnx) |
| RTMDet-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.onnx) |

</details>


### **注意:**
 - 所有模型均使用COCO train2017作为训练集，在COCO val2017上验证精度，模型前带*表示训练更新中。
 - 具体精度和速度细节请查看[PP-YOLOE](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/ppyoloe),[YOLOX](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolox),[YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5),[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7)，**其中YOLOv5,YOLOv6,YOLOv7评估并未采用`multi_label`形式**。
- 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，**不包含数据预处理和模型输出后处理(NMS)的耗时**。测试采用**单卡Tesla T4 GPU，batch size=1**，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考各自模型主页。
- **统计FLOPs(G)和Params(M)**，首先安装[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), `pip install paddleslim`，然后设置[runtime.yml](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/runtime.yml)里`print_flops: True`和`print_params: True`，并且注意确保是**单尺度**下如640x640，**打印的是MACs，FLOPs=2*MACs**。
 - 各模型导出后的权重以及ONNX，分为**带(w)**和**不带(wo)**后处理NMS，都提供了下载链接，请参考各自模型主页下载。`w_nms`表示**带NMS后处理**，可以直接使用预测出最终检测框结果如```python deploy/python/infer.py --model_dir=ppyoloe_crn_l_300e_coco_w_nms/ --image_file=demo/000000014439.jpg --device=GPU```；`wo_nms`表示**不带NMS后处理**，是**测速**时使用，如需预测出检测框结果需要找到**对应head中的后处理相关代码**并修改为如下：
 ```
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark for speed test
            # return pred_bboxes.sum(), pred_scores.sum() # 原先是这行，现在注释
            return pred_bboxes, pred_scores # 新加这行，表示保留进NMS前的原始结果
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
 ```
并重新导出，使用时再**另接自己写的NMS后处理**。
 - 基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)对YOLO系列模型进行量化训练，可以实现精度基本无损，速度普遍提升30%以上，具体请参照[模型自动化压缩工具ACT](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression)。


### [VOC](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc)

<details>
<summary> 基础模型 </summary>

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP(0.50,11point) | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :-----------: | :-------: | :-------: | :------: | :------------: | :---------------: | :------------------: |:-----------------: | :------: | :------: |
| YOLOv5-s        |  640     |    16     |   60e    |     3.2   |  80.3 |  7.24  | 16.54 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_s_60e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolov5_s_60e_voc.yml) |
| YOLOv7-tiny     |  640     |    32     |   60e    |     2.6   |  80.2 |  6.23  | 6.90 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov7_tiny_60e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolov7_tiny_60e_voc.yml) |
| YOLOX-s         |  640     |    8      |   40e    |     3.0   |  82.9 |  9.0   |  26.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_s_40e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/yolox_s_40e_voc.yml) |
| PP-YOLOE+_s     |  640     |    8      |   30e    |     2.9   |  86.7 |  7.93  |  17.36 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_30e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/voc/ppyoloe_plus_crn_s_30e_voc.yml) |

</details>

**注意:**
  - VOC数据集训练的mAP为`mAP(IoU=0.5)`的结果，且评估未使用`multi_label`等trick；
  - 所有YOLO VOC模型均加载各自模型的COCO权重作为预训练，各个配置文件的配置均为默认使用8卡GPU，可作为自定义数据集设置参考，具体精度会因数据集而异；
  - YOLO检测模型建议**总`batch_size`至少大于`64`**去训练，如果资源不够请**换小模型**或**减小模型的输入尺度**，为了保障较高检测精度，**尽量不要尝试单卡训和总`batch_size`小于`64`训**；
  - Params(M)和FLOPs(G)均为训练时所测，YOLOv7没有s模型，故选用tiny模型；
  - TRT-FP16-Latency(ms)测速相关请查看各YOLO模型的config的主页；


## 使用指南

下载MS-COCO数据集，[官网](https://cocodataset.org)下载地址为: [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip), [test2017](http://images.cocodataset.org/zips/test2017.zip)。
PaddleDetection团队提供的下载链接为：[coco](https://bj.bcebos.com/v1/paddledet/data/coco.tar)(共约22G)和[test2017](https://bj.bcebos.com/v1/paddledet/data/cocotest2017.zip)，注意test2017可不下载，评估是使用的val2017。


### **一键运行全流程**

将以下命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。

```bash
model_name=ppyoloe # 可修改，如 yolov7
job_name=ppyoloe_plus_crn_l_300e_coco # 可修改，如 yolov7_tiny_300e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡）
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.直接预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5

# 4.导出模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# 5.部署预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速，加 “--run_mode=trt_fp16” 表示在TensorRT FP16模式下测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx测速
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
```

- 如果想切换模型，只要修改开头两行即可，如:
  ```
  model_name=yolov7
  job_name=yolov7_l_300e_coco
  ```
- 导出**onnx**，首先安装[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)，`pip install paddle2onnx`；
- **统计FLOPs(G)和Params(M)**，首先安装[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)，`pip install paddleslim`，然后设置[runtime.yml](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/runtime.yml)里`print_flops: True`和`print_params: True`，并且注意确保是**单尺度**下如640x640，**打印的是MACs，FLOPs=2*MACs**。


### 自定义数据集

#### 数据集准备：

1.自定义数据集的标注制作，请参考[DetAnnoTools](../tutorials/data/DetAnnoTools.md);

2.自定义数据集的训练准备，请参考[PrepareDataSet](../tutorials/PrepareDataSet.md)。


#### fintune训练：

除了更改数据集的路径外，训练一般推荐加载**对应模型的COCO预训练权重**去fintune，会更快收敛和达到更高精度，如：

```base
# 单卡fintune训练：
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# 多卡fintune训练：
python -m paddle.distributed.launch --log_dir=./log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

**注意:**
- fintune训练一般会提示head分类分支最后一层卷积的通道数没对应上，属于正常情况，是由于自定义数据集一般和COCO数据集种类数不一致；
- fintune训练一般epoch数可以设置更少，lr设置也更小点如1/10，最高精度可能出现在中间某个epoch；

#### 预测和导出：

使用自定义数据集预测和导出模型时，如果TestDataset数据集路径设置不正确会默认使用COCO 80类。
除了TestDataset数据集路径设置正确外，也可以自行修改和添加对应的label_list.txt文件(一行记录一个对应种类)，TestDataset中的anno_path也可设置为绝对路径，如：
```
TestDataset:
  !ImageFolder
    anno_path: label_list.txt # 如不使用dataset_dir，则anno_path即为相对于PaddleDetection主目录的相对路径
    # dataset_dir: dataset/my_coco # 如使用dataset_dir，则dataset_dir/anno_path作为新的anno_path
```
label_list.txt里的一行记录一个对应种类，如下所示：
```
person
vehicle
```
