English | [简体中文](README_cn.md)

# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

[FairMOT](https://arxiv.org/abs/2004.01888) is based on an Anchor Free detector Centernet, which overcomes the problem of anchor and feature misalignment in anchor based detection framework. The fusion of deep and shallow features enables the detection and ReID tasks to obtain the required features respectively. It also uses low dimensional ReID features. FairMOT is a simple baseline composed of two homogeneous branches propose to predict the pixel level target score and ReID features. It achieves the fairness between the two tasks and  obtains a higher level of real-time MOT performance.

<div align="center">
  <img src="../../../../docs/images/ht_fairmot.gif" width='800'/>
</div>

## Model Zoo

### FairMOT Results on HT-21 Training Set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  67.2 |  70.4  |   9403  |  124840  |  255007  |     -   | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml) |


### FairMOT Results on HT-21 Test Set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  58.2  |  61.3  |  13166   |  141872  |  197074 |    -     | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml) |

**Notes:**
 FairMOT used 8 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training FairMOT on 8 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608_headtracking21/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml
```


### 2. Evaluation

Evaluating the track performance of FairMOT on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=output/fairmot_dla34_30e_1088x608_headtracking21/model_final.pdparams
```


### 3. Inference

Inference a vidoe on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams
```

### 5. Using exported model for python inference

```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_headtracking21 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**Notes:**
The tracking model is used to predict the video, and does not support the prediction of a single image. The visualization video of the tracking results is saved by default. You can add `--save_mot_txts` to save the txt result file, or `--save_images` to save the visualization images.


## Citations
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```
