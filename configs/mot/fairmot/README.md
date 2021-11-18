English | [简体中文](README_cn.md)

# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

[FairMOT](https://arxiv.org/abs/2004.01888) is based on an Anchor Free detector Centernet, which overcomes the problem of anchor and feature misalignment in anchor based detection framework. The fusion of deep and shallow features enables the detection and ReID tasks to obtain the required features respectively. It also uses low dimensional ReID features. FairMOT is a simple baseline composed of two homogeneous branches propose to predict the pixel level target score and ReID features. It achieves the fairness between the two tasks and  obtains a higher level of real-time MOT performance.

## Model Zoo

### FairMOT Results on MOT-16 Training Set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |   544  |  3822  |  14095  |     -   |    -   |   -    |
| DLA-34         | 1088x608 |  83.2  |  83.1  |   499  |  3861  |  14223  |     -   | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  80.8  |  81.1  |  561  |  3643  | 16967 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [config](./fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  74.0  |  76.1  |  640  |  4989  | 23034 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [config](./fairmot_dla34_30e_576x320.yml) |


### FairMOT Results on MOT-16 Test Set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |    -   |    -   |   25.9   |    -   |   -    |
| DLA-34         | 1088x608 |  75.0  |  74.7  |  919   |  7934  |  36747 |    -     | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  73.0  |  72.6  |  977   |  7578  |  40601 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [config](./fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  69.9  |  70.2  |  1044   |  8869  |  44898 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [config](./fairmot_dla34_30e_576x320.yml) |

**Notes:**
 FairMOT DLA-34 used 2 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.


### FairMOT enhance model
### Results on MOT-16 Test Set
| backbone       | input shape |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.9  |  74.7  |  1021   |  11425  |  31475 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [config](./fairmot_enhance_dla34_60e_1088x608.yml) |

### Results on MOT-17 Test Set
| backbone       | input shape |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  download  | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.3  |  74.2  |  3270  |  29112  | 106749 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [config](./fairmot_enhance_dla34_60e_1088x608.yml) |

**Notes:**
 FairMOT enhance DLA-34 used 8 GPUs for training and mini-batch size as 16 on each GPU，and trained for 60 epoches. The crowdhuman dataset is added to the train-set during training.


### FairMOT light model
### Results on MOT-16 Test Set
| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  71.7  |  66.6  |  1340  |  8642  | 41592 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [config](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |

### Results on MOT-17 Test Set
| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  70.7  |  65.7  |  4281  |  22485  | 138468 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [config](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |
| HRNetV2-W18   | 864x480  |  70.3  |  65.8  |  4056  |  18927  | 144486 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_864x480.pdparams) | [config](./fairmot_hrnetv2_w18_dlafpn_30e_864x480.yml) |
| HRNetV2-W18   | 576x320  |  65.3  |  64.8  |  4137  |  28860  | 163017 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams) | [config](./fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) |

**Notes:**
 FairMOT HRNetV2-W18 used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches. Only ImageNet pre-train model is used, and the optimizer adopts Momentum. The crowdhuman dataset is added to the train-set during training.


## Getting Start

### 1. Training

Training FairMOT on 2 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```


### 2. Evaluation

Evaluating the track performance of FairMOT on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```
**Notes:**
 The default evaluation dataset is MOT-16 Train Set. If you want to change the evaluation dataset, please refer to the following code and modify `configs/datasets/mot.yml`：
```
EvalMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    data_root: MOT17/images/train
    keep_ori_im: False # set True if save visualization images or video
```

### 3. Inference

Inference a vidoe on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams
```

### 5. Using exported model for python inference

```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
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
