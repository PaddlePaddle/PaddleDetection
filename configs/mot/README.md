English | [简体中文](README_cn.md)

# MOT (Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442) is basicly the same with SORT but added a CNN model to extract features in image of human part bounded by a detector. We use JDE as detection model to generate boxes, and select `PCBPyramid` as the ReID model. We also support loading the boxes from saved detection result files. [JDE](https://arxiv.org/abs/1909.12605) is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. [FairMOT](https://arxiv.org/abs/2004.01888) focuses on accomplishing the detection and re-identification in a single network to improve the inference speed, presents a simple baseline which consists of two homogeneous branches to predict pixel-wise objectness scores and re-ID features. The achieved fairness between the two tasks allows FairMOT to obtain high levels of detection and tracking accuracy.
<div align="center">
  <img src="../../docs/images/mot16_jde.gif" width=500 />
</div>

## Model Zoo

### JDE on MOT-16 training set

| backbone           | input shape | MOTA | IDF1  |  IDS  |   FP  |  FN  |  FPS  | download | config |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  73.2  |  69.4  | 1320  |  6613  | 21629 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  70.1  |  65.4  | 1341  |  6454  | 25208 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.1  |  64.6  | 1357  |  7083  | 32312 |   -   |[model](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**Notes:**
 JDE used 8 GPUs for training and mini-batch size as 4 on each GPU, and trained for 30 epoches.

### DeepSORT on MOT-16 training set

| backbone  | input shape  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | Detector | ReID | config |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:---: | :---: | :---: |
| DarkNet53 | 1088x608 |  72.2  |  60.3  | 998  |  8055  | 21631 |  3.28 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**Notes:**
 DeepSORT does not need to train, only used for evaluation. Before DeepSORT evaluation, you should get detection results by a detection model first, here we use JDE, and then prepare them like this:
```
det_results_dir
   |——————MOT16-02.txt
   |——————MOT16-04.txt
   |——————MOT16-05.txt
   |——————MOT16-09.txt
   |——————MOT16-10.txt
   |——————MOT16-11.txt
   |——————MOT16-13.txt
```

### FairMOT Results on MOT-16 train set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    MT   |   ML  | download | config |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: |:-------: | :----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |  544  |  3822  | 14095 | -------- | ------ |
| DLA-34         | 1088x608 |  83.4  |  82.7  |  517  |  4077  | 13761 | [model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

### FairMOT Results on MOT-16 test set

| backbone       | input shape | MOTA | IDF1 |  IDS  |    MT   |    ML   | download | config |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |  44.7% |  15.9% | -------- | ------ |
| DLA-34         | 1088x608 |  74.7  |  72.8  |  1044  |  41.9% |  19.1% |[model](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |

**Notes:**

FairMOT used 2 GPUs for training and mini-batch size as 6 on each GPU, and trained for 30 epoches.

## Getting Start

### 1. Training

Training FairMOT on 2 GPUs with following command

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml &>fairmot_dla34_30e_1088x608.log 2>&1 &
```

### 2. Evaluation

Evaluating the track performance of FairMOT on val dataset in single GPU with following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final
```

## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}

@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
