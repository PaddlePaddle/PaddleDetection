[English](README_en.md) | 简体中文

# 移动端模型库


## 模型

PaddleDetection目前提供一系列针对移动应用进行优化的模型，主要支持以下结构:

| 骨干网络                 | 结构                   | 输入大小 | 图片/gpu <sup>[1](#gpu)</sup>  | 学习率策略    | Box AP | 下载 | PaddleLite模型下载 |
| :----------------------- | :------------------------ | :---: | :--------------------: | :------------ | :----: | :--- | :----------------- |
| MobileNetV3 Small        | SSDLite                   | 320   | 64                     | 400K (cosine) | 16.2   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small.pdparams) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_small.tar) |
| MobileNetV3 Small        | SSDLite Quant <sup>[2](#quant)</sup> | 320   | 64                     | 400K (cosine) | 15.4   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small_quant.tar) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_small_quant.tar) |
| MobileNetV3 Large        | SSDLite                   | 320   | 64                     | 400K (cosine) | 23.3   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large.pdparams) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_large.tar) |
| MobileNetV3 Large        | SSDLite Quant <sup>[2](#quant)</sup> | 320   | 64                     | 400K (cosine) | 22.6   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large_quant.tar) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_large_quant.tar) |
| MobileNetV3 Large w/ FPN | Cascade RCNN              | 320   | 2                      | 500k (cosine) | 25.0   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/cascade_rcnn_mobilenetv3_fpn_320.tar) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/cascade_rcnn_mobilenetv3_fpn_320.tar) |
| MobileNetV3 Large w/ FPN | Cascade RCNN              | 640   | 2                      | 500k (cosine) | 30.2   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/cascade_rcnn_mobilenetv3_fpn_640.tar) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/cascade_rcnn_mobilenetv3_fpn_640.tar) |
| MobileNetV3 Large        | YOLOv3                    | 320   | 8                      | 500K          | 27.1   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3.tar) |
| MobileNetV3 Large        | YOLOv3 Prune <sup>[3](#prune)</sup> | 320   | 8                      | -             | 24.6   | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/yolov3_mobilenet_v3_prune75875_FPGM_distillby_r34.pdparams) | [链接](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3_prune86_FPGM_320.tar) |

**注意**:

-   <a name="gpu">[1]</a> 模型统一使用8卡训练。
-   <a name="quant">[2]</a> 参考下面关于[SSDLite量化的说明](#SSDLite量化说明)。
-   <a name="prune">[3]</a> 参考下面关于[YOLO剪裁的说明](#YOLOv3剪裁说明)。


## 评测结果

-   模型使用 [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 2.6 (即将发布) 在下列平台上进行了测试
    -   Qualcomm Snapdragon 625
    -   Qualcomm Snapdragon 835
    -   Qualcomm Snapdragon 845
    -   Qualcomm Snapdragon 855
    -   HiSilicon Kirin 970
    -   HiSilicon Kirin 980

-   单CPU线程 (单位： ms)

|                  | SD625   | SD835   | SD845   | SD855   | Kirin 970 | Kirin 980 |
|------------------|---------|---------|---------|---------|-----------|-----------|
| SSDLite Large    | 289.071 | 134.408 | 91.933  | 48.2206 | 144.914   | 55.1186   |
| SSDLite Large Quant |  |  |   |  |    |    |
| SSDLite Small    | 122.932 | 57.1914 | 41.003  | 22.0694 | 61.5468   | 25.2106   |
| SSDLite Small Quant |  |  |  | |   |    |
| YOLOv3 baseline  | 1082.5  | 435.77  | 317.189 | 155.948 | 536.987   | 178.999   |
| YOLOv3 prune     | 253.98  | 131.279 | 89.4124 | 48.2856 | 122.732   | 55.8626   |
| Cascade RCNN 320 | 286.526 | 125.635 | 87.404  | 46.184  | 149.179   | 52.9994   |
| Cascade RCNN 640 | 1115.66 | 495.926 | 351.361 | 189.722 | 573.558   | 207.917   |

-   4 CPU线程 (单位： ms)

|                  | SD625   | SD835   | SD845   | SD855   | Kirin 970 | Kirin 980 |
|------------------|---------|---------|---------|---------|-----------|-----------|
| SSDLite Large    | 107.535 | 51.1382 | 34.6392 | 20.4978 | 50.5598   | 24.5318   |
| SSDLite Large Quant |  |  |   |  |    |    |
| SSDLite Small    | 51.5704 | 24.5156 | 18.5486 | 11.4218 | 24.9946   | 16.7158   |
| SSDLite Small Quant |  |  |  | |   |    |
| YOLOv3 baseline  | 413.486 | 184.248 | 133.624 | 75.7354 | 202.263   | 126.435   |
| YOLOv3 prune     | 98.5472 | 53.6228 | 34.4306 | 21.3112 | 44.0722   | 31.201    |
| Cascade RCNN 320 | 131.515 | 59.6026 | 39.4338 | 23.5802 | 58.5046   | 36.9486   |
| Cascade RCNN 640 | 473.083 | 224.543 | 156.205 | 100.686 | 231.108   | 138.391   |

## SSDLite量化说明

在SSDLite模型中我们采用完整量化训练的方式对模型进行训练，在8卡GPU下共训练40万轮，训练中将`res_conv1`与`se_block`固定不训练，执行指令为：

```shell
python slim/quantization/train.py --not_quant_pattern res_conv1 se_block \
        -c configs/ssd/ssdlite_mobilenet_v3_large.yml \
        --eval
```
更多量化教程请参考[模型量化压缩教程](../../docs/advanced_tutorials/slim/quantization/QUANTIZATION.md)

## YOLOv3剪裁说明

首先对YOLO检测头进行剪裁，然后再使用 YOLOv3-ResNet34 作为teacher网络对剪裁后的模型进行蒸馏, teacher网络在COCO上的mAP为31.4 (输入大小320\*320).

可以使用如下两种方式进行剪裁:

-   固定比例剪裁, 整体剪裁率是86%

    ```shell
    --pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights" \
    --pruned_ratios="0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.875,0.875,0.875,0.875,0.875,0.875"
    ```
-   使用 [FPGM](https://arxiv.org/abs/1811.00250) 算法剪裁:

    ```shell
    --prune_criterion=geometry_median
    ```


## 敬请关注后续发布

-   [ ] 更多模型
-   [ ] 量化模型
