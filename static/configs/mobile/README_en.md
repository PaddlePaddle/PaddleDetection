English | [简体中文](README.md)

# Mobile Model Zoo


## Models

This directory contains models optimized for mobile applications, at present the following models included:

| Backbone                 | Architecture              | Input | Image/gpu <sup>[1](#gpu)</sup> | Lr schd       | Box AP | Download | PaddleLite Model Download |
| :----------------------- | :------------------------ | :---: | :--------------------: | :------------ | :----: | :------- | :------------------------ |
| MobileNetV3 Small        | SSDLite                   | 320   | 64                     | 400K (cosine) | 16.2   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small.pdparam) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_small.tar) |
| MobileNetV3 Small        | SSDLite Quant <sup>[2](#quant)</sup> | 320   | 64                     | 400K (cosine) | 15.4   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_small_quant.tar) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_small_quant.tar) |
| MobileNetV3 Large        | SSDLite                   | 320   | 64                     | 400K (cosine) | 23.3   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large.pdparam) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_large.tar) |
| MobileNetV3 Large        | SSDLite Quant <sup>[2](#quant)</sup> | 320   | 64                     | 400K (cosine) | 22.6   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/ssdlite_mobilenet_v3_large_quant.tar) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_large_quant.tar) |
| MobileNetV3 Large w/ FPN | Cascade RCNN              | 320   | 2                      | 500k (cosine) | 25.0   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/cascade_rcnn_mobilenetv3_fpn_320.tar) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/cascade_rcnn_mobilenetv3_fpn_320.tar) |
| MobileNetV3 Large w/ FPN | Cascade RCNN              | 640   | 2                      | 500k (cosine) | 30.2   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/cascade_rcnn_mobilenetv3_fpn_640.tar) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/cascade_rcnn_mobilenetv3_fpn_640.tar) |
| MobileNetV3 Large        | YOLOv3                    | 320   | 8                      | 500K          | 27.1   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3.tar) |
| MobileNetV3 Large        | YOLOv3 Prune <sup>2</sup> | 320   | 8                      | -             | 24.6   | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/yolov3_mobilenet_v3_prune75875_FPGM_distillby_r34.pdparams) | [Link](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3_prune86_FPGM_320.tar) |

**Notes**:

-   <a name="gpu">[1]</a> All models are trained on 8 GPUs.
-   <a name="quant">[2]</a> See the note section on [SSDLite quantization](#Notes-on-SSDLite-quant)。
-   <a name="prune">[3]</a> See the note section on [how YOLO head is pruned](#Notes-on-YOLOv3-pruning).


## Benchmarks Results

-   Models are benched on following chipsets with [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 2.6 (to be released)
    -   Qualcomm Snapdragon 625
    -   Qualcomm Snapdragon 835
    -   Qualcomm Snapdragon 845
    -   Qualcomm Snapdragon 855
    -   HiSilicon Kirin 970
    -   HiSilicon Kirin 980

-   With 1 CPU thread (latency numbers are in ms)

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

-   With 4 CPU threads (latency numbers are in ms)

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


## Notes on SSDLite quantization

We use a complete quantitative training method to train the SSDLite model. It is trained for a total of 400,000 rounds with the 8-card GPU. We freeze `res_conv1` and `se_block`. The command used is listed bellow:

```shell
python slim/quantization/train.py --not_quant_pattern res_conv1 se_block \
                -c configs/ssd/ssdlite_mobilenet_v3_large.yml \
                --eval
```

For more quantization tutorials, please refer to [Model Quantization Compression Tutorial](../../docs/advanced_tutorials/slim/quantization/QUANTIZATION.md)

## Notes on YOLOv3 pruning

We pruned the YOLO-head and distill the pruned model with YOLOv3-ResNet34 as the teacher, which has a higher mAP on COCO (31.4 with 320\*320 input).

The following configurations can be used for pruning:

-   Prune with fixed ratio, overall prune ratios is 86%

    ```shell
    --pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights" \
    --pruned_ratios="0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.875,0.875,0.875,0.875,0.875,0.875"
    ```
-   Prune filters using [FPGM](https://arxiv.org/abs/1811.00250) algorithm:

    ```shell
    --prune_criterion=geometry_median
    ```


## Upcoming

-   [ ] More models configurations
-   [ ] Quantized models
