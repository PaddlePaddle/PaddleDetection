简体中文 | [English](README.md)

# OC_SORT (Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
[OC_SORT](https://arxiv.org/abs/2203.14360)(Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking)。此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练epoch数、NMS阈值设置等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。

## 模型库

### OC_SORT在MOT-17 half Val Set上结果

|  检测训练数据集      |  检测器     | 输入尺度  |  ReID  |  检测mAP  |  MOTA  |  IDF1  |  FPS | 配置文件 |
| :--------         | :-----      | :----:  | :----:|:------:  | :----: |:-----: |:----:|:----:   |
| MOT-17 half train | PP-YOLOE-l  | 640x640 | -     |  52.9    |  50.1  |  62.6  |   -    |[配置文件](./ocsort_ppyoloe.yml) |
| **mot17_ch**       | YOLOX-x    | 800x1440|   -   |  61.9    |  75.5  |  77.0  |   -    |[配置文件](./ocsort_yolox.yml) |

**注意:**
  - 模型权重下载链接在配置文件中的```det_weights```和```reid_weights```，运行验证的命令即可自动下载，OC_SORT默认不需要```reid_weights```权重。
  - **MOT17-half train**是MOT17的train序列(共7个)每个视频的前一半帧的图片和标注组成的数据集，而为了验证精度可以都用**MOT17-half val**数据集去评估，它是每个视频的后一半帧组成的，数据集可以从[此链接](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip)下载，并解压放在`dataset/mot/`文件夹下。
  - **mix_mot_ch**数据集，是MOT17、CrowdHuman组成的联合数据集，**mix_det**是MOT17、CrowdHuman、Cityscapes、ETHZ组成的联合数据集，数据集整理的格式和目录可以参考[此链接](https://github.com/ifzhang/ByteTrack#data-preparation)，最终放置于`dataset/mot/`目录下。为了验证精度可以都用**MOT17-half val**数据集去评估。
  - OC_SORT的训练是单独的检测器训练MOT数据集，推理是组装跟踪器去评估MOT指标，单独的检测模型也可以评估检测指标。
  - OC_SORT的导出部署，是单独导出检测模型，再组装跟踪器运行的，参照[PP-Tracking](../../../deploy/pptracking/python)。
  - OC_SORT是PP-Human和PP-Vehicle等Pipeline分析项目跟踪方向的主要方案，具体使用参照[Pipeline](../../../deploy/pipeline)和[MOT](../../../deploy/pipeline/docs/tutorials/pphuman_mot.md)。


## 快速开始

### 1. 训练
通过如下命令一键式启动训练和评估
```bash
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp
```

### 2. 评估
#### 2.1 评估检测效果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml
```

**注意:**
 - 评估检测使用的是```tools/eval.py```, 评估跟踪使用的是```tools/eval_mot.py```。

#### 2.2 评估跟踪效果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/ocsort/ocsort_ppyoloe.yml --scaled=True
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/ocsort/ocsort_yolox.yml --scaled=True
```
**注意:**
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE YOLOv3则为False，如果使用通用检测模型则为True, 默认值是False。
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置。

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 下载demo视频
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/mot17_demo.mp4

CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/ocsort/ocsort_yolox.yml --video_file=mot17_demo.mp4 --scaled=True --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


### 4. 导出预测模型

Step 1：导出检测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/yolox_x_24e_800x1440_mix_det/ --tracker_config=deploy/pptracking/python/tracker_config.yml --video_file=mot17_demo.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 运行前需要手动修改`tracker_config.yml`的跟踪器类型为`type: OCSORTTracker`。
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)或`--save_mot_txt_per_img`(对每张图片保存一个txt)表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。


## 引用
```
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
