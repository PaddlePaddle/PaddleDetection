简体中文 | [English](README.md)

# JDE (Towards Real-Time Multi-Object Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

[JDE](https://arxiv.org/abs/1909.12605)(Joint Detection and Embedding)是在一个单一的共享神经网络中同时学习目标检测任务和embedding任务，并同时输出检测结果和对应的外观embedding匹配的算法。JDE原论文是基于Anchor Base的YOLOv3检测器新增加一个ReID分支学习embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。

### PP-Tracking 实时多目标跟踪系统
此外，PaddleDetection还提供了[PP-Tracking](../../../deploy/pptracking/README.md)实时多目标跟踪系统。PP-Tracking是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

<div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205540305-457d48bf-e9ec-4f28-896c-64c870126e05.gif" width=500 />
</div>

## 模型库

### JDE在MOT-16 Training Set上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  下载链接  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53          | 1088x608 |  72.0  |  66.9  | 1397  |  7274  | 22209 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53          | 864x480 |  69.1  |  64.7  | 1539  |  7544  | 25046 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |  63.7  |  64.4  | 1310  |  6782  | 31964 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_576x320.yml) |


### JDE在MOT-16 Test Set上结果

| 骨干网络            |  输入尺寸  |  MOTA  |  IDF1 |  IDS  |  FP  |  FN  |  FPS  |  下载链接  | 配置文件 |
| :----------------- | :------- | :----: | :----: | :---: | :----: | :---: | :---: | :---: | :---: |
| DarkNet53(paper)   | 1088x608 |  64.4  |  55.8  | 1544  |    -   |   -   |   -   |   -   |   -   |
| DarkNet53          | 1088x608 |  64.6  |  58.5  | 1864  |  10550 | 52088 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53(paper)   | 864x480 |   62.1  |  56.9  | 1608  |    -   |   -   |   -   |   -   |   -   |
| DarkNet53          | 864x480 |   63.2  |  57.7  | 1966  |  10070 | 55081 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_864x480.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53          | 576x320 |   59.1  |  56.4  | 1911  |  10923 | 61789 |   -   |[下载链接](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_576x320.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mot/jde/jde_darknet53_30e_576x320.yml) |

**注意:**
 - JDE使用8个GPU进行训练，每个GPU上batch size为4，训练了30个epoch。

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
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams --video_file={your video name}.mp4  --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/jde/jde_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/jde_darknet53_30e_1088x608 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。


## 引用
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
