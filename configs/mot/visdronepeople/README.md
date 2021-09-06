简体中文
# 特色垂类跟踪模型

## 无人机视频下的行人的跟踪（VisDronePeople Tracking)
VisDrone是无人机视角拍摄的数据集，针对VisDrone2019中的数据集进行提取，抽取出class为pedestrain的数据，所有序列的帧速率都是30fps。
<div align="center">
  <img src='../../../docs/images/VisDronePerson_uav0000099_02109_v.gif' width='800'/>
</div>

## 模型库
### FairMOT在VisDrone_people Training Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34         | 1088x608 |  39.2  |  50.8  |  5810   |  78864  |  46018 |    -     | [下载链接]() | [配置文件](fairmot_dla34_30e_1088x608.yml) |

### FairMOT在VisDrone_people Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  24.8 |  58.7  |   57  |  7908  |  2533  |     -   | [下载链接]() | [配置文件](fairmot_dla34_30e_1088x608.yml) |

**注意:**
 FairMOT使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。

## 快速开始

### 1. 训练
使用2个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/visdronepeople/fairmot_dla34_30e_1088x608.yml
```
### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/visdronepeople/fairmot_dla34_30e_1088x608.yml -o weights=people_output/fairmot_dla34_30e_1088x608/model_final.pdparams
```

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
# dataset/mot/visdrone_mot16_format_pedestrain/images/train/uav0000099_02109_v
# 加小尾巴的时候需要将原先infer的数据全部删除
rm -rf output/mot_outputs 
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/visdronepeople/fairmot_dla34_30e_1088x608.yml -o weights=people_output/fairmot_dla34_30e_1088x608/model_final.pdparams --image_dir=dataset/mot/visdrone_mot16_format_pedestrain/images/train/uav0000099_02109_v/img1 --save_videos --draw_threshold=0.2
```
**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/visdronepeople/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/model_final.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_headtracking21 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。

## 引用
```
[1] A. Milan, L. Leal-Taixe, K. Schindler, D. Cremers, S. Roth, and I. Reid, "Multiple Object Tracking Benchmark 2016", https://motchallenge.net/results/MOT16/.

[2] E. Park, W. Liu, O. Russakovsky, J. Deng, F.-F. Li, and A. Berg, "Large Scale Visual Recognition Challenge 2017", http://imagenet.org/challenges/LSVRC/2017
```
