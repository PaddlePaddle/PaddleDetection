简体中文 | [English](YOLOSERIES_MODEL_en.md)

# [**YOLOSeries**](https://github.com/nemonameless/PaddleDetection_YOLOSeries)


## 内容
- [简介](#简介)
- [模型库](#模型库)
    - [PP-YOLOE](#PP-YOLOE)
    - [YOLOX](#YOLOX)
    - [YOLOv5](#YOLOv5)
    - [YOLOv6](#YOLOv6)
    - [YOLOv7](#YOLOv7)
- [使用指南](#使用指南)
    - [一键运行全流程](#一键运行全流程)
    - [自定义数据集](#自定义数据集)

## 简介

[**YOLOSeries**](https://github.com/nemonameless/PaddleDetection_YOLOSeries)是基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的YOLO系列模型库，**由PaddleDetection团队成员建设和维护**，支持`YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`等模型，其upstream为PaddleDetection的[develop](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)分支，并与PaddleDetection主代码库分支保持同步更新，包括github和gitee的代码，欢迎一起使用和建设！

## Updates!
* 【2022/09/21】精简代码库只保留主要的YOLO模型相关的代码(release/2.5 branch)；
* 【2022/09/19】支持[`YOLOv6`](configs/yolov6)新版，包括n/t/s/m/l模型；
* 【2022/08/23】发布`PaddleDetection_YOLOSeries`代码库: 支持`YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`MT-YOLOv6`,`YOLOv7`等YOLO模型，支持ConvNeXt骨干网络高精度版`PP-YOLOE`,`YOLOX`和`YOLOv5`等模型，支持PaddleSlim无损加速量化训练`PP-YOLOE`,`YOLOv5`,`MT-YOLOv6`和`YOLOv7`等模型，详情可阅读[此文章](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


**注意:**
 - 此代码库**推荐使用paddlepaddle-2.3.0以上的版本**，请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载对应适合版本，**其中develop分支代码请安装paddle develop版本，其余分支建议安装paddle 2.3.2版本**。
 - github链接为：https://github.com/nemonameless/PaddleDetection_YOLOSeries
 - gitee链接为：https://gitee.com/nemonameless/PaddleDetection_YOLOSeries
 - 提issue可以在此代码库的[issues](https://github.com/nemonameless/PaddleDetection_YOLOSeries/issues)页面中，也可以在[PaddleDetection issues](https://github.com/PaddlePaddle/PaddleDetection/issues)中，也欢迎提[PR](https://github.com/nemonameless/PaddleDetection_YOLOSeries/pulls)共同建设和维护。
 - [PP-YOLOE](configs/ppyoloe),[PP-YOLOE+](configs/ppyoloe),[PP-YOLO](configs/ppyolo),[PP-YOLOv2](configs/ppyolo),[YOLOv3](configs/yolov3)和[YOLOX](configs/yolox)等模型推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中使用，**会最先发布PP-YOLO系列特色检测模型的最新进展**。
 - [YOLOv5](configs/yolov5),[YOLOv7](configs/yolov7)和[YOLOv6](configs/yolov6)模型推荐在此代码库中使用，**由于GPL开源协议而不合入[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)主代码库**。



## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> 技术交流

- 如果你发现任何PaddleDetection存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues)给我们提issues。

- **欢迎加入PaddleDetection 微信用户群（扫码填写问卷即可入群）**
  - **入群福利 💎：获取PaddleDetection团队整理的重磅学习大礼包🎁**
    - 📊 福利一：获取飞桨联合业界企业整理的开源数据集
    - 👨‍🏫 福利二：获取PaddleDetection历次发版直播视频与最新直播咨询
    - 🗳 福利三：获取垂类场景预训练模型集合，包括工业、安防、交通等5+行业场景
    - 🗂 福利四：获取10+全流程产业实操范例，覆盖火灾烟雾检测、人流量计数等产业高频场景
  <div align="center">
  <img src="https://user-images.githubusercontent.com/34162360/177678712-4655747d-4290-4ad9-b7a1-4564a5418ac6.jpg"  width = "200" />  
  </div>


## 模型库


### [PP-YOLOE, PP-YOLOE+](configs/ppyoloe)

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| PP-YOLOE-s   |     640   |    32    |  400e    |    2.9    |       43.4        |        60.0         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml)                   |
| PP-YOLOE-s   |     640   |    32    |  300e    |    2.9    |       43.0        |        59.6         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m   |      640  |    28    |  300e    |    6.0    |       49.0        |        65.9         |   23.43   |  49.91   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l   |      640  |    20    |  300e    |    8.7    |       51.4        |        68.6         |   52.20   |  110.07 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x   |      640  |    16    |  300e    |    14.9   |       52.3        |        69.5         |   98.42   |  206.59  |[model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)    |
| PP-YOLOE-tiny ConvNeXt| 640 |    16      |   36e    | -   |       44.6        |        63.3         |   33.04   |  13.87 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_convnext_tiny_36e_coco.pdparams) | [config](../../configs/convnext/ppyoloe_convnext_tiny_36e_coco.yml) |
| **PP-YOLOE+_s**   |     640   |    8    |  80e    |    2.9    |     **43.7**    |      **60.6**     |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)                   |
| **PP-YOLOE+_m**   |      640  |    8    |  80e    |    6.0    |     **49.8**    |      **67.1**     |   23.43   |  49.91   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml)                   |
| **PP-YOLOE+_l**   |      640  |    8    |  80e    |    8.7    |     **52.9**    |      **70.1**     |   52.20   |  110.07 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml)                   |
| **PP-YOLOE+_x**   |      640  |    8    |  80e    |    14.9   |     **54.7**    |      **72.0**     |   98.42   |  206.59  |[model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams) | [config](../../configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml)                   |



#### 部署模型


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



### [YOLOX](../../configs/yolox)

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOX-nano     |  416     |    8      |   300e    |     2.3    |  26.1  |  42.0 |  0.91  |  1.08 | [model](https://paddledet.bj.bcebos.com/models/yolox_nano_300e_coco.pdparams) | [config](../../configs/yolox/yolox_nano_300e_coco.yml) |
| YOLOX-tiny     |  416     |    8      |   300e    |     2.8    |  32.9  |  50.4 |  5.06  |  6.45 | [model](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [config](../../configs/yolox/yolox_tiny_300e_coco.yml) |
| YOLOX-s        |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  9.0  |  26.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams) | [config](../../configs/yolox/yolox_s_300e_coco.yml) |
| YOLOX-m        |  640     |    8      |   300e    |     5.8    |  46.9  |  65.7 |  25.3  |  73.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_m_300e_coco.pdparams) | [config](../../configs/yolox/yolox_m_300e_coco.yml) |
| YOLOX-l        |  640     |    8      |   300e    |     9.3    |  50.1  |  68.8 |  54.2  |  155.6 | [model](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) | [config](../../configs/yolox/yolox_l_300e_coco.yml) |
| YOLOX-x        |  640     |    8      |   300e    |     16.6   |  **51.8**  |  **70.6** |  99.1  |  281.9 | [model](https://paddledet.bj.bcebos.com/models/yolox_x_300e_coco.pdparams) | [config](../../configs/yolox/yolox_x_300e_coco.yml) |
 YOLOX-cdn-tiny    |  416     |    8      |   300e    |     1.9    |  32.4  |  50.2 |  5.03 |  6.33  | [model](https://paddledet.bj.bcebos.com/models/yolox_cdn_tiny_300e_coco.pdparams) | [config](c../../onfigs/yolox/yolox_cdn_tiny_300e_coco.yml) |
| YOLOX-crn-s     |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  7.7  |  24.69 | [model](https://paddledet.bj.bcebos.com/models/yolox_crn_s_300e_coco.pdparams) | [config](../../configs/yolox/yolox_crn_s_300e_coco.yml) |
| YOLOX-s ConvNeXt|  640     |    8      |   36e     |     -      |  44.6  |  65.3 |  36.2 |  27.52 | [model](https://paddledet.bj.bcebos.com/models/yolox_convnext_s_36e_coco.pdparams) | [config](../../configs/convnext/yolox_convnext_s_36e_coco.yml) |


#### 部署模型

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOx-nano |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_nano_300e_coco_wo_nms.onnx) |
| YOLOx-tiny |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_tiny_300e_coco_wo_nms.onnx) |
| YOLOx-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_s_300e_coco_wo_nms.onnx) |
| YOLOx-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_m_300e_coco_wo_nms.onnx) |
| YOLOx-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_l_300e_coco_wo_nms.onnx) |
| YOLOx-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolox/yolox_x_300e_coco_wo_nms.onnx) |


### [YOLOv5](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5)

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-n        |  640     |    16     |   300e    |     2.6    |  28.0  | 45.7 |  1.87  | 4.52 | [model](https://paddledet.bj.bcebos.com/models/yolov5_n_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    8      |   300e    |     3.2    |  37.0  | 55.9 |  7.24  | 16.54 | [model](https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    5      |   300e    |     5.2    |  45.3  | 63.8 |  21.19  | 49.08 | [model](https://paddledet.bj.bcebos.com/models/yolov5_m_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    3      |   300e    |     7.9    |  48.6  | 66.9 |  46.56  | 109.32 | [model](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    2      |   300e    |     13.7    |  **50.6**  | **68.7** |  86.75  | 205.92 | [model](https://paddledet.bj.bcebos.com/models/yolov5_x_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_x_300e_coco.yml) |
| YOLOv5-s ConvNeXt|  640    |    8      |   36e     |     -      |  42.4  |  65.3  |  34.54 |  17.96 | [model](https://paddledet.bj.bcebos.com/models/yolov5_convnext_s_36e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5/yolov5_convnext_s_36e_coco.yml) |

#### 部署模型


| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv5-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.onnx) |
| YOLOv5-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.onnx) |
| YOLOv5-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.onnx) |
| YOLOv5-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.onnx) |
| YOLOv5-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.onnx) |



### [YOLOv6](configs/yolov6)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *YOLOv6-n       |  416     |    32      |   400e    |     1.0    |  31.1 |    45.3 |  4.74  | 5.16 |[model](https://paddledet.bj.bcebos.com/models/yolov6_n_416_400e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_n_416_400e_coco.yml) |
| *YOLOv6-n       |  640     |    32      |   400e    |     1.3    |  36.1 |    51.9 |  4.74  | 12.21 |[model](https://paddledet.bj.bcebos.com/models/yolov6_n_400e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_n_400e_coco.yml) |
| *YOLOv6-t       |  640     |    32      |   400e    |     2.1    |  40.7 |    57.4 |  10.63  | 27.29 |[model](https://paddledet.bj.bcebos.com/models/yolov6_t_400e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_t_400e_coco.yml) |
| *YOLOv6-s       |  640     |    32      |   400e    |     2.6    |  43.4 |    60.5 |  18.87  | 48.35 |[model](https://paddledet.bj.bcebos.com/models/yolov6_s_400e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_s_400e_coco.yml) |
| *YOLOv6-m       |  640     |    32      |   300e    |     5.0    |  49.0 |    66.5 |  37.17  | 88.82 |[model](https://paddledet.bj.bcebos.com/models/yolov6_m_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_m_300e_coco.yml) |
| *YOLOv6-l       |  640     |    32      |   300e    |     7.9    |  51.0 |    68.9 |  63.54  | 155.89 |[model](https://paddledet.bj.bcebos.com/models/yolov6_l_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_l_300e_coco.yml) |
| *YOLOv6-l-silu  |  640     |    32      |   300e    |     9.6    |  51.7 |    69.6 |  58.59  | 142.66 |[model](https://paddledet.bj.bcebos.com/models/yolov6_l_silu_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6/yolov6_l_silu_300e_coco.yml) |


#### 部署模型


| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| yolov6-n |  416   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_416_400e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_416_400e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_416_400e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_416_400e_coco_wo_nms.onnx) |
| yolov6-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_400e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_400e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_400e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_n_400e_coco_wo_nms.onnx) |
| yolov6-t |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_t_400e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_t_400e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_t_400e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_t_400e_coco_wo_nms.onnx) |
| yolov6-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_400e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_400e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_400e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_s_400e_coco_wo_nms.onnx) |
| yolov6-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_m_300e_coco_wo_nms.onnx) |
| yolov6-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_300e_coco_wo_nms.onnx) |
| yolov6-l-silu |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_silu_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_silu_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_silu_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov6/yolov6_l_silu_300e_coco_wo_nms.onnx) |


### [YOLOv7](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7)

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv7-L        |  640     |    32      |   300e    |     7.4     |  51.0  | 70.2 |  37.62  | 106.08 |[model](https://paddledet.bj.bcebos.com/models/yolov7_l_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7_l_300e_coco.yml) |
| *YOLOv7-X        |  640     |    32      |   300e    |     12.2    |  53.0  | 70.8 |  71.34  | 190.08 | [model](https://paddledet.bj.bcebos.com/models/yolov7_x_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7_x_300e_coco.yml) |
| *YOLOv7P6-W6     |  1280    |    16      |   300e    |     25.5    |  54.4  | 71.8 |  70.43  | 360.26 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_w6_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7p6_w6_300e_coco.yml) |
| *YOLOv7P6-E6     |  1280    |    10      |   300e    |     31.1    |  55.7  | 73.0 |  97.25  | 515.4 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7p6_e6_300e_coco.yml) |
| *YOLOv7P6-D6     |  1280    |    8      |   300e    |     37.4    | 56.1  | 73.3 |  133.81  | 702.92 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_d6_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7p6_d6_300e_coco.yml) |
| *YOLOv7P6-E6E    |  1280    |    6      |   300e    |     48.7    |  56.5  | 73.7 |  151.76  | 843.52 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6e_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7p6_e6e_300e_coco.yml) |
| YOLOv7-tiny     |  640     |    32      |   300e    |     -   |  37.3 | 54.5 |  6.23  | 6.90 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7_tiny_300e_coco.yml) |
| YOLOv7-tiny     |  416     |    32      |   300e    |     -    | 33.3 | 49.5 |  6.23  | 2.91 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_416_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7_tiny_416_300e_coco.yml) |
| YOLOv7-tiny     |  320     |    32      |   300e    |     -    | 29.1 | 43.8 |  6.23  | 1.73 |[model](https://paddledet.bj.bcebos.com/models/yolov7_tiny_320_300e_coco.pdparams) | [config](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7/yolov7_tiny_320_300e_coco.yml) |


#### 部署模型


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


### **注意:**
 - 所有模型均使用COCO train2017作为训练集，在COCO val2017上验证精度，模型前带*表示训练更新中。
 - 具体精度和速度细节请查看[PP-YOLOE](../../configs/ppyoloe),[YOLOX](../../configs/yolox),[YOLOv5](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5),[YOLOv6](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs),[YOLOv7](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7)。
- 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用单卡V100，batch size=1，测试环境为**paddlepaddle-2.3.0**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考各自模型主页。
- **统计参数量Params(M)**，可以将以下代码插入[trainer.py](https://github.com/nemonameless/PaddleDetection_YOLOSeries/blob/develop/ppdet/engine/trainer.py#L150)。
  ```python
  params = sum([
      p.numel() for n, p in self.model.named_parameters()
      if all([x not in n for x in ['_mean', '_variance']])
  ]) # exclude BatchNorm running status
  print('Params: ', params / 1e6)
  ```

- **统计FLOPs(G)**，首先安装[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), `pip install paddleslim`，然后设置[runtime.yml](../../configs/runtime.yml)里`print_flops: True`，并且注意确保是**单尺度**下如640x640，**打印的是MACs，FLOPs=2*MACs**。
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

 - [PP-YOLOE](../../configs/ppyoloe),[PP-YOLOE+](../../configs/ppyoloe),[YOLOv3](../../configs/yolov3)和[YOLOX](../../configs/yolox)推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)里使用，会最先发布**PP-YOLO系列特色检测模型的最新进展**。
 - [YOLOv5](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov5),[YOLOv7](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov7)和[YOLOv6](https://github.com/nemonameless/PaddleDetection_YOLOSeries/tree/develop/configs/yolov6)由于GPL协议而不合入[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)主代码库。
 - **paddlepaddle版本推荐使用2.3.0版本以上**。



## 使用指南

下载MS-COCO数据集，[官网](https://cocodataset.org)下载地址为: [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip), [test2017](http://images.cocodataset.org/zips/test2017.zip)。
PaddleDetection团队提供的下载链接为：[coco](https://bj.bcebos.com/v1/paddledet/data/coco.tar)(共约22G)和[test2017](https://bj.bcebos.com/v1/paddledet/data/cocotest2017.zip)，注意test2017可不下载，评估是使用的val2017。


### **一键运行全流程**
```
model_type=ppyoloe # 可修改，如 yolov7
job_name=ppyoloe_crn_l_300e_coco # 可修改，如 yolov7_l_300e_coco

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡）
# CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.直接预测
CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5

# 4.导出模型
CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# 5.部署预测
CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速
CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx测速
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16

```

**注意:**
- 将以上命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。
- 如果想切换模型，只要修改开头两行即可，如:
  ```
  model_type=yolov7
  job_name=yolov7_l_300e_coco
  ```

- **统计参数量Params(M)**，可以将以下代码插入[trainer.py](https://github.com/nemonameless/PaddleDetection_YOLOSeries/blob/develop/ppdet/engine/trainer.py#L150)。
  ```python
  params = sum([
      p.numel() for n, p in self.model.named_parameters()
      if all([x not in n for x in ['_mean', '_variance']])
  ]) # exclude BatchNorm running status
  print('Params: ', params / 1e6)
  ```

- **统计FLOPs(G)**，首先安装[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), `pip install paddleslim`，然后设置[runtime.yml](../../configs/runtime.yml)里`print_flops: True`，并且注意确保是**单尺度**下如640x640，**打印的是MACs，FLOPs=2*MACs**。

### 自定义数据集

#### 数据集准备：

1.自定义数据集的标注制作，请参考[DetAnnoTools](../tutorials/data/DetAnnoTools.md);

2.自定义数据集的训练准备，请参考[PrepareDataSet](../tutorials/PrepareDataSet.md)。


#### fintune训练：

除了更改数据集的路径外，训练一般推荐加载**对应模型的COCO预训练权重**去fintune，会更快收敛和达到更高精度，如：

```base
# 单卡fintune训练：
# CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# 多卡fintune训练：
python3.7 -m paddle.distributed.launch --log_dir=./log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
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
