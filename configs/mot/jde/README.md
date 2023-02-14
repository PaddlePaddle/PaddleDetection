English | [简体中文](README_cn.md)

# JDE (Towards Real-Time Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

- [JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding) learns the object detection task and appearance embedding task simutaneously in a shared neural network. And the detection results and the corresponding embeddings are also outputed at the same time. JDE original paper is based on an Anchor Base detector YOLOv3, adding a new ReID branch to learn embeddings. The training process is constructed as a multi-task learning problem, taking into account both accuracy and speed.

### PP-Tracking real-time MOT system
In addition, PaddleDetection also provides [PP-Tracking](../../../deploy/pptracking/README.md) real-time multi-object tracking system.
PP-Tracking is the first open source real-time Multi-Object Tracking system, and it is based on PaddlePaddle deep learning framework. It has rich models, wide application and high efficiency deployment.

PP-Tracking supports two paradigms: single camera tracking (MOT) and multi-camera tracking (MTMCT). Aiming at the difficulties and pain points of actual business, PP-Tracking provides various MOT functions and applications such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. The deployment method supports API and GUI visual interface, and the deployment language supports Python and C++, The deployment platform environment supports Linux, NVIDIA Jetson, etc.

### AI studio public project tutorial
PP-tracking provides an AI studio public project tutorial. Please refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).

<div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205540305-457d48bf-e9ec-4f28-896c-64c870126e05.gif" width=500 />
</div>

## Model Zoo

### JDE Results on MOT-16 Training Set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  72.0  |  66.9  | 1397  |  7274  | 22209 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  69.1  |  64.7  | 1539  |  7544  | 25046 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.7  |  64.4  | 1310  |  6782  | 31964 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

### JDE Results on MOT-16 Test Set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53(paper)   | 1088x608 |  64.4  |  55.8  | 1544  |    -    |   -   |   -   |   -  |   -   |
| DarkNet53          | 1088x608 |  64.6  |  58.5  | 1864  |  10550 | 52088 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53(paper)   | 864x480 |   62.1  |  56.9  | 1608  |    -    |   -   |   -   |   -  |   -   |
| DarkNet53          | 864x480 |   63.2  |  57.7  | 1966  |  10070  | 55081 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |   59.1  |  56.4  | 1911  |  10923  | 61789  |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**Notes:**
 - JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training JDE on 8 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./jde_darknet53_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml
```

### 2. Evaluation

Evaluating the track performance of JDE on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=output/jde_darknet53_30e_1088x608/model_final.pdparams
```
**Notes:**
 - The default evaluation dataset is MOT-16 Train Set. If you want to change the evaluation dataset, please refer to the following code and modify `configs/datasets/mot.yml`：
  ```
  EvalMOTDataset:
    !MOTImageFolder
      dataset_dir: dataset/mot
      data_root: MOT17/images/train
      keep_ori_im: False # set True if save visualization images or video
  ```
 - Tracking results will be saved in `{output_dir}/mot_results/`, and every sequence has one txt file, each line of the txt file is `frame,id,x1,y1,w,h,score,-1,-1,-1`, and you can set `{output_dir}` by `--output_dir`.

### 3. Inference

Inference a video on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 - Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams
```

### 5. Using exported model for python inference

```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/jde_darknet53_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**Notes:**
 - The tracking model is used to predict the video, and does not support the prediction of a single image. The visualization video of the tracking results is saved by default. You can add `--save_mot_txts` to save the txt result file, or `--save_images` to save the visualization images.
 - Each line of the tracking results txt file is `frame,id,x1,y1,w,h,score,-1,-1,-1`.


## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
