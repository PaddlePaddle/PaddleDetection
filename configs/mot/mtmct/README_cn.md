English | [简体中文](README_cn.md)

# MTMCT (Multi-Target Multi-Camera Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction
MTMCT (Multi-Target Multi-Camera Tracking) is 

## Model Zoo
### DeepSORT在 AIC21 MTMCT(CityFlow) 车辆跨境跟踪数据集Test集上的结果

|  检测器       |  输入尺度      |  ReID     |  场景       |  MOTA  |  IDF1  |  FPS  | 配置文件 |
|  :-----      | :-----       | :----:     | :----:     | :----: |:-----: |:----: |:----:  |
| PicoDet      | 1088x608     | PPLCNet    | MultiCam   |  7.38  |  11.19  |   -   |[配置文件](./deepsort_picodet_pplcnet_aic21mtmct_vehicle.yml) |
| PPYOLOv2     | 1088x608     | PPLCNet    | MultiCam    |  -  |  -  |   -   |[配置文件](./deepsort_ppyolov2_pplcnet.yml) |

**Notes:**


## Getting Start

### 1. Training
Training PicoDet_l_896 detector on 8 GPUs with following command
```bash
python -m paddle.distributed.launch --log_dir=./mcfairmot_dla34_30e_1088x608_visdrone/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml
```

### 2. Evaluation
Evaluating the track performance of MCFairMOT on val dataset in single GPU with following commands:
```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=output/mcfairmot_dla34_30e_1088x608_visdrone/model_final.pdparams
```
**Notes:**
 The default evaluation dataset is VisDrone2019 MOT val-set. If you want to change the evaluation dataset, please refer to the following code and modify `configs/datasets/mcmot.yml`：
```
EvalMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    data_root: your_dataset/images/val
    keep_ori_im: False # set True if save visualization images or video
```

### 3. Inference
Inference a vidoe on single GPU with following command:
```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams
```

### 5. Using exported model for python inference
```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/mcfairmot_dla34_30e_1088x608_visdrone --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**Notes:**
The tracking model is used to predict the video, and does not support the prediction of a single image. The visualization video of the tracking results is saved by default. You can add `--save_mot_txts` to save the txt result file, or `--save_images` to save the visualization images.


## Citations
```
@InProceedings{Tang19CityFlow,
author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},
title = {CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019},
pages = {8797–8806}
}
```
