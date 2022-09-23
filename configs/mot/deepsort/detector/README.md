English | [简体中文](README_cn.md)

# Detector for DeepSORT

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) is composed of a detector and a ReID model in series. The configs of several common detectors are provided here as a reference. Note that different training dataset, backbone, input size, training epochs and NMS threshold will lead to differences in model accuracy and performance. Please adapt according to your needs.

## Model Zoo
### Results on MOT17-half dataset
| Backbone        | Model           | input size  | lr schedule |  FPS          | Box AP  |  download    | config  |
| :-------------- | :-------------  | :--------:  | :---------: | :-----------: | :-----: | :----------: | :-----: |
| DarkNet-53      | YOLOv3          |   608X608   |   40e      |      ----     |  42.7   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/yolov3_darknet53_40e_608x608_mot17half.pdparams)  | [config](./yolov3_darknet53_40e_608x608_mot17half.yml) |
| ResNet50-vd     | PPYOLOv2        |   640x640   |   365e      |      ----     |  46.8   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyolov2_r50vd_dcn_365e_640x640_mot17half.pdparams)  | [config](./ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml) |
| CSPResNet       | PPYOLOe         |   640x640   |   36e       |      ----     |  52.9   | [download](https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyoloe_crn_l_36e_640x640_mot17half.pdparams)     | [config](./ppyoloe_crn_l_36e_640x640_mot17half.yml)    |

**Notes:**
  - The above models are trained with **MOT17-half train** set, it can be downloaded from this [link](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip).
  - **MOT17-half train** set is a dataset composed of pictures and labels of the first half frame of each video in MOT17 Train dataset (7 sequences in total). **MOT17-half val set** is used for evaluation, which is composed of the second half frame of each video. They can be downloaded from this [link](https://paddledet.bj.bcebos.com/data/mot/mot17half/annotations.zip). Download and unzip it in the `dataset/mot/MOT17/images/`folder.
  - YOLOv3 is trained with the same pedestrian dataset as `configs/pphuman/pedestrian_yolov3/pedestrian_yolov3_darknet.yml`, which is not open yet.
  - For pedestrian tracking, please use pedestrian detector combined with pedestrian ReID model. For vehicle tracking, please use vehicle detector combined with vehicle ReID model.
  - High quality detected boxes are required for DeepSORT tracking, so the post-processing settings such as NMS threshold of these models are different from those in pure detection tasks.

## Quick Start

Start the training and evaluation with the following command
```bash
job_name=ppyoloe_crn_l_36e_640x640_mot17half
config=configs/mot/deepsort/detector/${job_name}.yml
log_dir=log_dir/${job_name}
# 1. training
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp --fleet
# 2. evaluation
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/${job_name}.pdparams
```
