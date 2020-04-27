# Practical Mobile-side detection method base on PaddleDetection

Mobile-side models are provided base on following architecture:

1. YOLOv3
2. Cascade Faster RCNN
3. SSD

## YOLOv3 mobile-side model

Mobile-side model based on YOLOv3 is a pruned model of YOLOv3-MobileNetv3, we pruned the YOLO-head of YOLOv3-MobileNetv3 and distill the pruned model by YOLOv3-ResNet34, which has a higher mAP on COCO as 31.4(input shape as 320\*320). For pruning, configurations as as follows:

1. pruning YOLO-head with following configuration and the FLOPS pruned ratios is 86%.

```
--pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights" \
--pruned_ratios="0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.875,0.875,0.875,0.875,0.875,0.875"
```

2. pruning YOLO-head by [Filter Pruning via Geometric Median](https://arxiv.org/abs/1811.00250), use FPGM algorithm by setting:

```
--prune_criterion=geometry_median
```

### Model Zoo

| Backbone         | prune method |     GFLOPs    | Model size(MB) | input shape |     teacher model     |   Box AP   | SD845 latency |                      download                          |
| :----------------| :----------: | :-----------: | :------------: | :---------: | :-------------------: | :--------: | :-----------: |:-----------------------------------------------------: |
| MobileNetv3      |   baseline   | 4.93          | 90.23          |     320     |           -           | 27.1       |     319ms     | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) |
| MobileNetv3      |    prune     | 0.66(-86.57%) | 16.21(-82.03%) |     320     | YOLOv3-ResNet34(31.4) | 24.6(-2.5) |      91ms     | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3_mobilenetv3_prune75875_FPGM_distillby_r34.pdparams) |

**NOTE：** `baseline` is the YOLOv3-MobileNetv3 base model, `prune` is the pruned model from YOLOv3-MobileNetv3 and pruned by configurations above, `Box AP` is test by `320*320` as input shape in both two models, and `latency` is test on Snapdragon845 with single thread. The pruned model is 2.5 times faster than the base model when the `Box AP` only decreased by 2.5.
