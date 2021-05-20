简体中文 | [English](README.md)

# JDE (Joint Detection and Embedding)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

[JDE](https://arxiv.org/abs/1909.12605) (Joint Detection and Embedding)是一个快速高性能多目标跟踪器，它是在共享神经网络中同时学习目标检测任务和外观嵌入任务的。
<div align="center">
  <img src="../../../docs/images/mot16_jde.gif" width=500 />
</div>

## 模型库

### JDE在MOT-16 train集上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  检测模型  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  73.2  |  69.3  | 1351  |  6591  | 21625 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  70.1  |  65.2  | 1328  |  6441  | 25187 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.2  |  64.5  | 1308  |  7011  | 32252 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**注意:**
 JDE使用8个GPU进行训练，每个GPU上batch size为4，训练了30个epoch。

## 快速开始

### 1. 训练

使用8GPU通过如下命令一键式启动训练

```bash
python -m paddle.distributed.launch --log_dir=./jde_darknet53_30e_1088x608/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml
```

### 2. 评估

使用8GPU通过如下命令一键式启动评估

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=output/jde_darknet53_30e_1088x608/model_final.pdparams
```

## 引用
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
