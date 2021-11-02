English | [简体中文](README_cn.md)

# Object Detection for DeepSORT

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) is composed of a detector and a ReID model in series. The configs of several common detectors are provided here as a reference. Note that different training dataset, backbone, input size and training epochs will lead to differences in model accuracy and performance. Please adapt according to your needs.

## Model Zoo
### Results on MOT17-half dataset
| Backbone        | Model           | input size  | lr schedule |  FPS          | Box AP  |  download    | config  |
| :-------------- | :-------------  | :--------:  | :---------: | :-----------: | :-----: | :----------: | :-----: |
| ResNet50-vd     | PPYOLOv2        |   640x640   |   365e      |      ----     |  45.0   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/ppyolov2_r50vd_dcn_365e_640x640_mot17half.pdparams)  | [config](./ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml) |
| ResNet50-FPN    | Faster R-CNN    |   1333x800  |   1x        |      ----     |  42.9   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/faster_rcnn_r50_fpn_2x_1333x800_mot17half.pdparams)  | [config](./faster_rcnn_r50_fpn_2x_1333x800_mot17half.yml) |
| ESNet           | PicoDet         |    896x896  |   300e      |      ----     |  40.4   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/picodet_l_esnet_300e_640x640_mot17half.pdparams)     | [config](./picodet_l_esnet_300e_640x640_mot17half.yml)    |

**Notes:**
  The above model is trained with MOT17-half Train set, which is a dataset composed of pictures and labels of the first half frame of each video in MOT17 Train dataset (7 sequences in total). In order to verify the accuracy, it is evaluated with MOT17-half Val set, which is composed of the second half frame of each video.
  For pedestrian tracking, please use pedestrian detector combined with pedestrian ReID model. For vehicle tracking, please use vehicle detector combined with vehicle ReID model.
