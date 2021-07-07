English | [简体中文](README_cn.md)

# JDE (Towards Real-Time Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

- [JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding) learns the object detection task and appearance embedding task simutaneously in a shared neural network. And the detection results and the corresponding embeddings are also outputed at the same time. JDE original paper is based on an Anchor Base detector YOLOv3, adding a new ReID branch to learn embeddings. The training process is constructed as a multi-task learning problem, taking into account both accuracy and speed.

<div align="center">
  <img src="../../../docs/images/mot16_jde.gif" width=500 />
</div>

## Model Zoo

### JDE on MOT-16 Training Set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  72.0  |  66.9  | 1397  |  7274  | 22209 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  69.1  |  64.7  | 1539  |  7544  | 25046 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.7  |  64.4  | 1310  |  6782  | 31964 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

### JDE on MOT-16 Test Set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53(paper)   | 1088x608 |  64.4  |  55.8  | 1544  |    -    |   -   |   -   |   -  |   -   |
| DarkNet53          | 1088x608 |  64.6  |  58.5  | 1864  |  10550 | 52088 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53(paper)   | 864x480 |   62.1  |  56.9  | 1608  |    -    |   -   |   -   |   -  |   -   |
| DarkNet53          | 864x480 |   63.2  |  57.7  | 1966  |  10070  | 55081 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |   59.1  |  56.4  | 1911  |  10923  | 61789  |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**Notes:**
 JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

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
 The default evaluation dataset is MOT-16 Train Set. If you want to change the evaluation dataset, please refer to the following code and modify `configs/datasets/mot.yml`：
```
EvalMOTDataset:
  !MOTImageFolder
    task: MOT17_train
    dataset_dir: dataset/mot
    data_root: MOT17/images/train
    keep_ori_im: False # set True if save visualization images or video
```

### 3. Inference

Inference a vidoe on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
