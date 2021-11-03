简体中文 | [English](README.md)

# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

[FairMOT](https://arxiv.org/abs/2004.01888)以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

## 模型库

### FairMOT在MOT-16 Training Set上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34(paper)  | 1088x608 |  83.3  |  81.9  |  544  |  3822  | 14095 |    -     |   -   |   -   |
| DLA-34         | 1088x608 |  83.2  |  83.1  |  499  |  3861  | 14223 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  80.8  |  81.1  |  561  |  3643  | 16967 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [配置文件](./fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  74.0  |  76.1  |  640  |  4989  | 23034 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [配置文件](./fairmot_dla34_30e_576x320.yml) |

### FairMOT在MOT-16 Test Set上结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34(paper)  | 1088x608 |  74.9  |  72.8  |  1074  |    -   |    -   |   25.9   |    -   |   -    |
| DLA-34         | 1088x608 |  75.0  |  74.7  |  919   |  7934  |  36747 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34         | 864x480 |  73.0  |  72.6  |  977   |  7578  |  40601 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_864x480.pdparams) | [配置文件](./fairmot_dla34_30e_864x480.yml) |
| DLA-34         | 576x320 |  69.9  |  70.2  |  1044   |  8869  |  44898 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_576x320.pdparams) | [配置文件](./fairmot_dla34_30e_576x320.yml) |

**注意:**
 FairMOT DLA-34均使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


### FairMOT enhance模型
### 在MOT-16 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.9  |  74.7  |  1021   |  11425  |  31475 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |

### 在MOT-17 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.3  |  74.2  |  3270  |  29112  | 106749 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |

**注意:**
 FairMOT enhance DLA-34使用8个GPU进行训练，每个GPU上batch size为16，训练60个epoch，并且训练集中加入了crowdhuman数据集一起参与训练。


### FairMOT轻量级模型
### 在MOT-16 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  71.7  |  66.6  |  1340  |  8642  | 41592 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |

### 在MOT-17 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| HRNetV2-W18   | 1088x608 |  70.7  |  65.7  |  4281  |  22485  | 138468 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608.yml) |
| HRNetV2-W18   | 864x480  |  70.3  |  65.8  |  4056  |  18927  | 144486 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_864x480.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_864x480.yml) |
| HRNetV2-W18   | 576x320  |  65.3  |  64.8  |  4137  |  28860  | 163017 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) |

**注意:**
 FairMOT HRNetV2-W18均使用8个GPU进行训练，每个GPU上batch size为4，训练30个epoch，使用的ImageNet预训练，优化器策略采用的是Momentum，并且训练集中加入了crowdhuman数据集一起参与训练。


## 快速开始

### 1. 训练

使用2个GPU通过如下命令一键式启动训练

```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```

### 2. 评估

使用单张GPU通过如下命令一键式启动评估

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```
**注意:**
 默认评估的是MOT-16 Train Set数据集, 如需换评估数据集可参照以下代码修改`configs/datasets/mot.yml`：
```
EvalMOTDataset:
  !MOTImageFolder
    dataset_dir: dataset/mot
    data_root: MOT17/images/train
    keep_ori_im: False # set True if save visualization images or video
```

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```

**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。

## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```
