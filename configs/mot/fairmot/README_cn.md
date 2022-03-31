简体中文 | [English](README.md)

# FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

[FairMOT](https://arxiv.org/abs/2004.01888)以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

### PP-Tracking 实时多目标跟踪系统
此外，PaddleDetection还提供了[PP-Tracking](../../../deploy/pptracking/README.md)实时多目标跟踪系统。PP-Tracking是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

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
 - FairMOT DLA-34均使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。


### FairMOT enhance模型
### 在MOT-16 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.9  |  74.7  |  1021   |  11425  |  31475 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |
| HarDNet-85     | 1088x608 |  75.0  |  70.0  |  1050   |  11837  |  32774 | -        |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_hardnet85_30e_1088x608.pdparams) | [配置文件](./fairmot_enhance_hardnet85_30e_1088x608.yml) |

### 在MOT-17 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  75.3  |  74.2  |  3270  |  29112  | 106749 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_dla34_60e_1088x608.pdparams) | [配置文件](./fairmot_enhance_dla34_60e_1088x608.yml) |
| HarDNet-85     | 1088x608 |  74.7  |  70.7  |  3210  |  29790  | 109914 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_enhance_hardnet85_30e_1088x608.pdparams) | [配置文件](./fairmot_enhance_hardnet85_30e_1088x608.yml) |

**注意:**
 - FairMOT enhance模型均使用8个GPU进行训练，训练集中加入了crowdhuman数据集一起参与训练。
 - FairMOT enhance DLA-34 每个GPU上batch size为16，训练60个epoch。
 - FairMOT enhance HarDNet-85 每个GPU上batch size为10，训练30个epoch。

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
 - FairMOT HRNetV2-W18均使用8个GPU进行训练，每个GPU上batch size为4，训练30个epoch，使用的ImageNet预训练，优化器策略采用的是Momentum，并且训练集中加入了crowdhuman数据集一起参与训练。

### FairMOT + BYTETracker

### 在MOT-17 Half上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  69.1  |  72.8  |  299  |  1957  | 14412 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608.yml) |
| DLA-34 + BYTETracker| 1088x608 |  70.3 |  73.2  |  234  |  2176  | 13598 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_bytetracker.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_bytetracker.yml) |


**注意:**
 - FairMOT模型此处是ablation study的配置，使用的训练集是原先MIX的5个数据集(Caltech,CUHKSYSU,PRW,Cityscapes,ETHZ)加上MOT17 Train的前一半，且使用是预训练权重是CenterNet的COCO预训练权重，验证是在MOT17 Train的后一半上测的。
 - BYTETracker应用到PaddleDetection的其他FairMOT模型，只需要更改对应的config文件里的tracker部分为如下所示：
 ```
 JDETracker:
  use_byte: True
  match_thres: 0.8
  conf_thres: 0.4
  low_conf_thres: 0.2
 ```

### FairMOT迁移学习模型

### 在GMOT-40的airplane子集上的结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| DLA-34         | 1088x608 |  96.6  |  94.7  |   19   |  300   | 466    |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_airplane.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_airplane.yml) |

**注意:**
 - 此模型数据集是GMOT-40的airplane类别抽离出来的子集，PaddleDetection团队整理后的下载链接为: ```wget https://bj.bcebos.com/v1/paddledet/data/mot/airplane.zip```，下载解压存放于 ```dataset/mot```目录下，并将其中的```airplane.train```复制存放于```dataset/mot/image_lists```。
 - FairMOT模型此处训练是采用行人FairMOT训好的模型作为预训练权重，使用的训练集是airplane全集共4个视频序列，验证也是在全集上测的。
 - 应用到其他物体的跟踪，需要更改对应的config文件里的tracker部分的```min_box_area```和```vertical_ratio```，如下所示：
 ```
JDETracker:
  conf_thres: 0.4
  tracked_thresh: 0.4
  metric_type: cosine
  min_box_area: 0 # 200 for pedestrian
  vertical_ratio: 0 # 1.6 for pedestrian
 ```

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
 - 默认评估的是MOT-16 Train Set数据集, 如需换评估数据集可参照以下代码修改`configs/datasets/mot.yml`：
  ```
  EvalMOTDataset:
    !MOTImageFolder
      dataset_dir: dataset/mot
      data_root: MOT17/images/train
      keep_ori_im: False # set True if save visualization images or video
  ```
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置。

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。

### 6. 用导出的跟踪和关键点模型Python联合预测

```bash
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```
**注意:**
 - 关键点模型导出教程请参考`configs/keypoint/README.md`。


## 引用
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
