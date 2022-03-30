English | [简体中文](README_cn.md)

# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

[FairMOT](https://arxiv.org/abs/2004.01888) is based on an Anchor Free detector Centernet, which overcomes the problem of anchor and feature misalignment in anchor based detection framework. The fusion of deep and shallow features enables the detection and ReID tasks to obtain the required features respectively. It also uses low dimensional ReID features. FairMOT is a simple baseline composed of two homogeneous branches propose to predict the pixel level target score and ReID features. It achieves the fairness between the two tasks and  obtains a higher level of real-time MOT performance.

### PP-Tracking real-time MOT system
In addition, PaddleDetection also provides [PP-Tracking](../../../deploy/pptracking/README.md) real-time multi-object tracking system.
PP-Tracking is the first open source real-time Multi-Object Tracking system, and it is based on PaddlePaddle deep learning framework. It has rich models, wide application and high efficiency deployment.

PP-Tracking supports two paradigms: single camera tracking (MOT) and multi-camera tracking (MTMCT). Aiming at the difficulties and pain points of actual business, PP-Tracking provides various MOT functions and applications such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. The deployment method supports API and GUI visual interface, and the deployment language supports Python and C++, The deployment platform environment supports Linux, NVIDIA Jetson, etc.

### AI studio public project tutorial
PP-tracking provides an AI studio public project tutorial. Please refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).


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
 - FairMOT DLA-34 used 2 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.


### FairMOT enhance model
### Results on MOT-16 Test Set
| backbone       | input shape |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.9  |  74.7  |  1021   |  11425  |  31475 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [config](./fairmot_enhance_dla34_60e_1088x608.yml) |
| HarDNet-85     | 1088x608 |  75.0  |  70.0  |  1050   |  11837  |  32774 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_hardnet85_30e_1088x608.pdparams) | [config](./fairmot_enhance_hardnet85_30e_1088x608.yml) |

### Results on MOT-17 Test Set
| backbone       | input shape |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  download  | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.3  |  74.2  |  3270  |  29112  | 106749 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [config](./fairmot_enhance_dla34_60e_1088x608.yml) |
| HarDNet-85     | 1088x608 |  74.7  |  70.7  |  3210  |  29790  | 109914 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_hardnet85_30e_1088x608.pdparams) | [config](./fairmot_enhance_hardnet85_30e_1088x608.yml) |

**Notes:**
 - FairMOT enhance used 8 GPUs for training, and the crowdhuman dataset is added to the train-set during training.
 - For FairMOT enhance DLA-34 the batch size is 16 on each GPU，and trained for 60 epoches.
 - For FairMOT enhance HarDNet-85 the batch size is 10 on each GPU，and trained for 30 epoches.

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
 - FairMOT HRNetV2-W18 used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches. Only ImageNet pre-train model is used, and the optimizer adopts Momentum. The crowdhuman dataset is added to the train-set during training.

### FairMOT + BYTETracker

### Results on MOT-17 Half Set
| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  69.1  |  72.8  |  299  |  1957  | 14412 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34 + BYTETracker| 1088x608 |  70.3 |  73.2  |  234  |  2176  | 13598 |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bytetracker.pdparams) | [config](./fairmot_dla34_30e_1088x608_bytetracker.yml) |

**Notes:**
 - FairMOT here is for ablation study, the training dataset is the 5 datasets of MIX(Caltech,CUHKSYSU,PRW,Cityscapes,ETHZ) and the first half of MOT17 Train, and the pretrain weights is CenterNet COCO model, the evaluation is on the second half of MOT17 Train.
 - BYTETracker adapt to other FairMOT models of PaddleDetection, you can modify the tracker of the config like this:
 ```
 JDETracker:
  use_byte: True
  match_thres: 0.8
  conf_thres: 0.4
  low_conf_thres: 0.2
 ```

### Fairmot transfer learning model

### Results on GMOT-40 airplane subset
| backbone       | input shape | MOTA | IDF1 |  IDS  |    FP   |   FN   |    FPS    | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  96.6  |  94.7  |   19   |  300   | 466    |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_airplane.pdparams) | [config](./fairmot_dla34_30e_1088x608_airplane.yml) |

**Note:**
 - The dataset of this model is a subset of airport category extracted from GMOT-40 dataset. The download link provided by the PaddleDetection team is```wget https://bj.bcebos.com/v1/paddledet/data/mot/airplane.zip```, unzip and store it in the ```dataset/mot```, and then copy the ```airplane.train``` to ```dataset/mot/image_lists```.
 - FairMOT model here uses the pedestrian FairMOT trained model for pre- training weights. The train-set used is the complete set of airplane, with a total of 4 video sequences, and it also used for evaluation.
- When applied to the tracking other objects, you should modify ```min_box_area``` and ```vertical_ratio``` of the tracker in the corresponding config file,  like this：
 ```
JDETracker:
  conf_thres: 0.4
  tracked_thresh: 0.4
  metric_type: cosine
  min_box_area: 0 # 200 for pedestrian
  vertical_ratio: 0 # 1.6 for pedestrian
 ```


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

Inference a vidoe on single GPU with following command:

```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 - Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams
```

### 5. Using exported model for python inference

```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**Notes:**
 - The tracking model is used to predict the video, and does not support the prediction of a single image. The visualization video of the tracking results is saved by default. You can add `--save_mot_txts` to save the txt result file, or `--save_images` to save the visualization images.
 - Each line of the tracking results txt file is `frame,id,x1,y1,w,h,score,-1,-1,-1`.


### 6. Using exported MOT and keypoint model for unite python inference

```bash
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```
**Notes:**
 - Keypoint model export tutorial: `configs/keypoint/README.md`.


## Citations
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
@article{shao2018crowdhuman,
  title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
  author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:1805.00123},
  year={2018}
}
```
