[English](README.md) | 简体中文
# 特色垂类跟踪模型

## 人头跟踪（Head Tracking)

现有行人跟踪器对高人群密度场景表现不佳，人头跟踪更适用于密集场景的跟踪。
[HT-21](https://motchallenge.net/data/Head_Tracking_21)是一个高人群密度拥挤场景的人头跟踪数据集，场景包括不同的光线和环境条件下的拥挤的室内和室外场景，所有序列的帧速率都是25fps。
<div align="center">
  <img src="../../../docs/images/ht_fairmot.gif" width='800'/>
</div>

## 模型库
### FairMOT在HT-21 Training Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  64.7 |  69.0  |   8533  |  148817  |  234970  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_headtracking21.yml) |
| HRNetv2-W18    | 1088x608 |  57.2 |  58.4  |   30950 |  188260  |  256580  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_headtracking21.pdparams) | [配置文件](./fairmot_hrnetv2_w18_dlafpn_30e_1088x608_headtracking21.yml) |


### FairMOT在HT-21 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34         | 1088x608 |  60.8  |  62.8  |  12781   |  118109  |  198896 |    -     | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_headtracking21.yml) |
| HRNetv2-W18    | 1088x608 |  41.2  |  47.1  |  48809   |  241683  |  204346 |    -     | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams) | [配置文件](./fairmot_dla34_30e_1088x608_headtracking21.yml) |

**注意:**
 - FairMOT DLA-34使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。目前MOTA精度位于MOT官网[Head Tracking 21](https://motchallenge.net/results/Head_Tracking_21)榜单榜首。
 - FairMOT HRNetv2-W18使用4个GPU进行训练，每个GPU上batch size为8，训练30个epoch。

## 快速开始

### 1. 训练
使用2个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608_headtracking21/ --gpus 0,1 tools/train.py -c configs/mot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml
```

### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=output/fairmot_dla34_30e_1088x608_headtracking21/model_final.pdparams
```

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams --video_file={your video name}.mp4  --save_videos
```
**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/headtracking21/fairmot_dla34_30e_1088x608_headtracking21.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608_headtracking21.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_headtracking21 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。

## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
@InProceedings{Sundararaman_2021_CVPR,
    author    = {Sundararaman, Ramana and De Almeida Braga, Cedric and Marchand, Eric and Pettre, Julien},
    title     = {Tracking Pedestrian Heads in Dense Crowd},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3865-3875}
}
```
