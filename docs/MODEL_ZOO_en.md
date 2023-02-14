# Model Zoos and Baselines

# Content
- [Basic Settings](#Basic-Settings)
    - [Test Environment](#Test-Environment)
    - [General Settings](#General-Settings)
    - [Training strategy](#Training-strategy)
    - [ImageNet pretraining model](#ImageNet-pretraining-model)
- [Baseline](#Baseline)
    - [Object Detection](#Object-Detection)
    - [Instance Segmentation](#Instance-Segmentation)
    - [PaddleYOLO](#PaddleYOLO)
    - [Face Detection](#Face-Detection)
    - [Rotated Object detection](#Rotated-Object-detection)
    - [KeyPoint Detection](#KeyPoint-Detection)
    - [Multi Object Tracking](#Multi-Object-Tracking)

# Basic Settings

## Test Environment

- Python 3.7
- PaddlePaddle Daily version
- CUDA 10.1
- cuDNN 7.5
- NCCL 2.4.8

## General Settings

- All models were trained and tested in the COCO17 dataset.
- The codes of [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov5),[YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov6),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov7) and [YOLOv8](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov8) can be found in [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO). Note that **the LICENSE of PaddleYOLO is GPL 3.0**.
- Unless special instructions, all the ResNet backbone network using [ResNet-B](https://arxiv.org/pdf/1812.01187) structure.
- **Inference time (FPS)**: The reasoning time was calculated on a Tesla V100 GPU by `tools/eval.py` testing all validation sets in FPS (number of pictures/second). CuDNN version is 7.5, including data loading, network forward execution and post-processing, and Batch size is 1.

## Training strategy

- We adopt and [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules) in the same training strategy.
- 1x strategy indicates that when the total batch size is 8, the initial learning rate is 0.01, and the learning rate decreases by 10 times after 8 epoch and 11 epoch, respectively, and the final training is 12 epoch.
- 2x strategy is twice as much as strategy 1x, and the learning rate adjustment position of epochs is twice as much as strategy 1x.

## ImageNet pretraining model
Paddle provides a skeleton network pretraining model based on ImageNet. All pre-training models were trained by standard Imagenet 1K dataset. ResNet and MobileNet are high-precision pre-training models obtained by cosine learning rate adjustment strategy or SSLD knowledge distillation training. Model details are available at [PaddleClas](https://github.com/PaddlePaddle/PaddleClas).


# Baseline

## Object Detection

### Faster R-CNN

Please refer to [Faster R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/faster_rcnn/)

### YOLOv3

Please refer to [YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/)

### PP-YOLOE/PP-YOLOE+

Please refer to [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe/)

### PP-YOLO/PP-YOLOv2

Please refer to [PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/)

### PicoDet

Please refer to [PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet)

### RetinaNet

Please refer to [RetinaNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/retinanet/)

### Cascade R-CNN

Please refer to [Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/cascade_rcnn)

### SSD/SSDLite

Please refer to [SSD](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ssd/)

### FCOS

Please refer to [FCOS](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/fcos/)

### CenterNet

Please refer to [CenterNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/centernet/)

### TTFNet/PAFNet

Please refer to [TTFNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/)

### Group Normalization

Please refer to [Group Normalization](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gn/)

### Deformable ConvNets v2

Please refer to [Deformable ConvNets v2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/dcn/)

### HRNets

Please refer to [HRNets](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/hrnet/)

### Res2Net

Please refer to [Res2Net](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/res2net/)

### ConvNeXt

Please refer to [ConvNeXt](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/convnext/)

### GFL

Please refer to [GFL](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl)

### TOOD

Please refer to [TOOD](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/tood)

### PSS-DET(RCNN-Enhance)

Please refer to [PSS-DET](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rcnn_enhance)

### DETR

Please refer to [DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/detr)

### Deformable DETR

Please refer to [Deformable DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/deformable_detr)

### Sparse R-CNN

Please refer to [Sparse R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/sparse_rcnn)

###  Vision Transformer

Please refer to [Vision Transformer](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/vitdet)

### DINO

Please refer to [DINO](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/dino)

### YOLOX

Please refer to [YOLOX](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolox)

### YOLOF

Please refer to [YOLOF](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolof)


## Instance-Segmentation

### Mask R-CNN

Please refer to [Mask R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mask_rcnn/)

### Cascade R-CNN

Please refer to [Cascade R-CNN](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/cascade_rcnn)

### SOLOv2

Please refer to [SOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/solov2/)

### QueryInst

Please refer to [QueryInst](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/queryinst)


## [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)

Please refer to [Model Zoo for PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/docs/MODEL_ZOO_en.md)

### YOLOv5

Please refer to [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov5)

### YOLOv6(v3.0)

Please refer to [YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov6)

### YOLOv7

Please refer to [YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov7)

### YOLOv8

Please refer to [YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov8)

### RTMDet

Please refer to [RTMDet](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/rtmdet)


## Face Detection

Please refer to [Model Zoo for Face Detection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/face_detection)

### BlazeFace

Please refer to [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/face_detection/)


## Rotated Object detection

Please refer to [Model Zoo for Rotated Object Detection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate)

### PP-YOLOE-R

Please refer to [PP-YOLOE-R](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r)

### FCOSR

Please refer to [FCOSR](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr)

### S2ANet

Please refer to [S2ANet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet)


## KeyPoint Detection

Please refer to [Model Zoo for KeyPoint Detection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint)

### PP-TinyPose

Please refer to [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/tiny_pose)

### HRNet

Please refer to [HRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/hrnet)

### Lite-HRNet

Please refer to [Lite-HRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/lite_hrnet)

### HigherHRNet

Please refer to [HigherHRNet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/keypoint/higherhrnet)


## Multi-Object Tracking

Please refer to [Model Zoo for Multi-Object Tracking](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot)

### DeepSORT

Please refer to [DeepSORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/deepsort)

### ByteTrack

Please refer to [ByteTrack](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/bytetrack)

### OC-SORT

Please refer to [OC-SORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/ocsort)

### BoT-SORT

Please refer to [BoT-SORT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/botsort)

### CenterTrack

Please refer to [CenterTrack](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/centertrack)

### FairMOT/MC-FairMOT

Please refer to [FairMOT](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/fairmot)

### JDE

Please refer to [JDE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde)
