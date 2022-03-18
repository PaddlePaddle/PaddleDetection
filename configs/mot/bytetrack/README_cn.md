简体中文 | [English](README.md)

# ByteTrack (ByteTrack: Multi-Object Tracking by Associating Every Detection Box)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
[ByteTrack](https://arxiv.org/abs/2110.06864)(ByteTrack: Multi-Object Tracking by Associating Every Detection Box) 通过关联每个检测框来跟踪，而不仅是关联高分的检测框。对于低分数检测框会利用它们与轨迹片段的相似性来恢复真实对象并过滤掉背景检测框。此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练epoch数、NMS阈值设置等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。

## 模型库

### ByteTrack在MOT-17 half Val Set上结果

|  检测训练数据集      |  检测器     | 输入尺度  |  ReID  |  检测mAP  |  MOTA  |  IDF1  |  FPS | 配置文件 |
| :--------         | :-----      | :----:  | :----:|:------:  | :----: |:-----: |:----:|:----:   |
| MOT-17 half train | YOLOv3      | 608x608 | -     |  42.7    |  49.5  |  54.8  |   -    |[配置文件](./bytetrack_yolov3.yml) |
| MOT-17 half train | PPYOLOe     | 640x640 | -     |  52.9    |  50.4  |  59.7  |   -    |[配置文件](./bytetrack_ppyoloe.yml) |
| MOT-17 half train | PPYOLOe     | 640x640 |PPLCNet|  52.9    |  51.7  |  58.8  |   -    |[配置文件](./bytetrack_ppyoloe_pplcnet.yml) |

**注意:**
- 模型权重下载链接在配置文件中的```det_weights```和```reid_weights```，运行验证的命令即可自动下载。
- ByteTrack的训练是单独的检测器训练MOT数据集，推理是组装跟踪器去评估MOT指标，单独的检测模型也可以评估检测指标。
- ByteTrack的导出部署，是单独导出检测模型，再组装跟踪器运行的，参照[PP-Tracking](../../../deploy/pptracking/python/README.md)。


## 快速开始

### 1. 训练
通过如下命令一键式启动训练和评估
```bash
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp --fleet
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
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/bytetrack/bytetrack_yolov3.yml --scaled=True
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/bytetrack/bytetrack_ppyoloe.yml --scaled=True
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/bytetrack/bytetrack_ppyoloe_pplcnet.yml --scaled=True
```
**注意:**
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE YOLOv3则为False，如果使用通用检测模型则为True, 默认值是False。
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置。

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/bytetrack/bytetrack_ppyoloe.yml --video_file={your video name}.mp4 --scaled=True --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


### 4. 导出预测模型

Step 1：导出检测模型
```bash
# 导出PPYOLe行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
```

Step 2：导出ReID模型(可选步骤，默认不需要)
```bash
# 导出PPLCNet ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pplcnet.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams
```

### 4. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/ppyoloe_crn_l_36e_640x640_mot17half/ --tracker_config=tracker_config.yml --video_file={your video name}.mp4 --device=GPU --scaled=True --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)或`--save_mot_txt_per_img`(对每张图片保存一个txt)表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


## 引用
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
