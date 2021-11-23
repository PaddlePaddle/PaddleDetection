# Model Libraries and Baselines

## Test Environment

- Python 3.7
- PaddlePaddle Daily version
- CUDA 10.1
- cuDNN 7.5
- NCCL 2.4.8

## General Settings

- All models were trained and tested in the COCO17 dataset.
- Unless special instructions, all the ResNet backbone network using [ResNet-B](https://arxiv.org/pdf/1812.01187) structure.
- **Inference time (FPS)**: The reasoning time was calculated on a Tesla V100 GPU by `tools/eval.py` testing all validation sets in FPS (number of pictures/second). CuDNN version is 7.5, including data loading, network forward execution and post-processing, and Batch size is 1.

## Training strategy

- We adopt and [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules) in the same training strategy.
- 1x strategy indicates that when the total batch size is 8, the initial learning rate is 0.01, and the learning rate decreases by 10 times after 8 epoch and 11 epoch, respectively, and the final training is 12 epoch.
- 2X strategy is twice as much as strategy 1X, and the learning rate adjustment position is twice as much as strategy 1X.

## ImageNet pretraining model
Paddle provides a skeleton network pretraining model based on ImageNet. All pre-training models were trained by standard Imagenet 1K dataset. Res Net and Mobile Net are high-precision pre-training models obtained by cosine learning rate adjustment strategy or SSLD knowledge distillation training. Model details are available at [PaddleClas](https://github.com/PaddlePaddle/PaddleClas).


## Baseline

### Faster R-CNN

Please refer to[Faster R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/faster_rcnn/)

### Mask R-CNN

Please refer to[Mask R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn/)

### Cascade R-CNN

Please refer to[Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/cascade_rcnn)

### YOLOv3

Please refer to[YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/)

### SSD

Please refer to[SSD](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd/)

### FCOS

Please refer to[FCOS](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/fcos/)

### SOLOv2

Please refer to[SOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/solov2/)

### PP-YOLO

Please refer to[PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/)

### TTFNet

请参考[TTFNet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ttfnet/)

### Group Normalization

Please refer to[Group Normalization](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gn/)

### Deformable ConvNets v2

Please refer to[Deformable ConvNets v2](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/dcn/)

### HRNets

Please refer to[HRNets](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/hrnet/)

### Res2Net

Please refer to[Res2Net](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/res2net/)

### GFL

Please refer to[GFL](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gfl)

### PicoDet

Please refer to[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet)


## Rotating frame detection

### S2ANet

Please refer to[S2ANet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/dota/)

## KeyPoint Detection

### PP-TinyPose

Please refer to [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/keypoint/tiny_pose)

## HRNet

Please refer to [HRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/keypoint/hrnet)

## HigherHRNet

Please refer to [HigherHRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/keypoint/higherhrnet)

## Multi-Object Tracking

### DeepSort

Please refer to [DeepSort](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort)

### JDE

Please refer to [JDE](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde)

### fairmot 

Please refer to [FairMot](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot)