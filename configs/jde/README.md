English | [简体中文](README_cn.md)

# JDE (Towards-Realtime-MOT)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

[Joint Detection and Embedding](https://arxiv.org/abs/1909.12605)(JDE) is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network。
JDE reached 64.4 MOTA on MOT16-tesing datatset.
<div align="center">
  <img src="../../../docs/images/mot16_jde.gif" width=500 />
</div>

## Model Zoo

### JDE on MOT-16 training set

| backbone           | input shape  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | download  | config |
| :-----------------| :------- | :----: | :----: | :---: | :----: | :---: | :---: |:---: | :---: |
| DarkNet53(paper)  | 1088x608 |  74.8  |  67.3  | 1189  |  5558  | 21505 |  22.2 | ---- | ---- |
| DarkNet53(paper)  | 864x480  |  70.8  |  65.8  | 1279  |  5653  | 25806 |  30.3 | ---- | ---- |
| DarkNet53(paper)  | 576x320  |  63.7  |  63.3  | 1307  |  6657  | 32794 |  37.9 | ---- | ---- |
| DarkNet53         | 1088x608 |    -   |    -   |   -   |    -   |   -   |   -   |[model](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53         | 864x480  |    -   |    -   |   -   |    -   |   -   |   -   |[model](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53         | 576x320  |    -   |    -   |   -   |    -   |   -   |   -   |[model](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_576x320.yml) |


**Notes:**
 JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training JDE on 8 GPUs with following command(all commands should be run under PaddleDetection dygraph directory, the input shape is 1088x608 as default)

```bash
python -m paddle.distributed.launch --log_dir=./jde_darknet53_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/jde/jde_darknet53_30e_1088x608.yml &>jde_1088x608.log 2>&1 &
```


### 2. Evaluation

Evaluating the detector module of JDE on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/jde/jde_darknet53_30e_1088x608.yml -o weights=output/jde_darknet53_30e_1088x608/model_final
```

Evaluating the re-id module of JDE on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/jde/jde_darknet53_30e_1088x608_testemb.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/jde/jde_darknet53_30e_1088x608_testemb.yml -o weights=output/jde_darknet53_30e_1088x608/model_final
```

Evaluating the track performance of JDE on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/jde/jde_darknet53_30e_1088x608_track.yml -o weights=output/jde_darknet53_30e_1088x608/model_final
```

### 3. Inference

Inference images in single GPU with following commands, use `--infer_img` to inference a single image and `--infer_dir` to inference all images in the directory.

```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py configs/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams --infer_img=../demo/000000014439_640x640.jpg

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py configs/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams --infer_dir=../demo
```

Inference vidoe in single GPU with following commands.

```bash
# inference on video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py configs/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams --video_file=../demo/input.mp4

```
## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
