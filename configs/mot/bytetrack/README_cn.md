简体中文 | [English](README.md)

# ByteTrack (ByteTrack: Multi-Object Tracking by Associating Every Detection Box)

## 内容
- [简介](#简介)
- [模型库](#模型库)
    - [行人跟踪](#行人跟踪)
    - [人头跟踪](#人头跟踪)
- [多类别适配](#多类别适配)
- [快速开始](#快速开始)
- [引用](#引用)


## 简介
[ByteTrack](https://arxiv.org/abs/2110.06864)(ByteTrack: Multi-Object Tracking by Associating Every Detection Box) 通过关联每个检测框来跟踪，而不仅是关联高分的检测框。对于低分数检测框会利用它们与轨迹片段的相似性来恢复真实对象并过滤掉背景检测框。此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练epoch数、NMS阈值设置等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。


## 模型库

### 行人跟踪

#### 基于不同检测器的ByteTrack在 MOT-17 half Val Set 上的结果

|  检测训练数据集      |  检测器     | 输入尺度  |  ReID  |  检测mAP(0.5:0.95)  |  MOTA  |  IDF1  |  FPS | 配置文件 |
| :--------         | :-----      | :----:  | :----:|:------:  | :----: |:-----: |:----:|:----:   |
| MOT-17 half train | YOLOv3      | 608x608 | -     |  42.7    |  49.5  |  54.8  |   -    |[配置文件](./bytetrack_yolov3.yml) |
| MOT-17 half train | PP-YOLOE-l  | 640x640 | -     |  52.9    |  50.4  |  59.7  |   -    |[配置文件](./bytetrack_ppyoloe.yml) |
| MOT-17 half train | PP-YOLOE-l  | 640x640 |PPLCNet|  52.9    |  51.7  |  58.8  |   -    |[配置文件](./bytetrack_ppyoloe_pplcnet.yml) |
| **mix_mot_ch** | YOLOX-x     | 800x1440|   -   |  61.9    |  77.3  |  71.6  |   -    |[配置文件](./bytetrack_yolox.yml) |
| **mix_det** | YOLOX-x     | 800x1440|   -   |  65.4    |  84.5  |  77.4  |   -    |[配置文件](./bytetrack_yolox.yml) |

**注意:**
  - 检测任务相关配置和文档请查看[detector](detector/)。
  - 模型权重下载链接在配置文件中的```det_weights```和```reid_weights```，运行```tools/eval_mot.py```评估的命令即可自动下载，```reid_weights```若为None则表示不需要使用。
  - **ByteTrack默认不使用ReID权重**，如需使用ReID权重，可以参考 [bytetrack_ppyoloe_pplcnet.yml](./bytetrack_ppyoloe_pplcnet.yml)，如需**更换ReID权重，可改动其中的`reid_weights: `为自己的权重路径**。
  - **MOT17-half train**是MOT17的train序列(共7个)每个视频的前一半帧的图片和标注组成的数据集，而为了验证精度可以都用**MOT17-half val**数据集去评估，它是每个视频的后一半帧组成的，数据集可以从[此链接](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip)下载，并解压放在`dataset/mot/`文件夹下。
  - **mix_mot_ch**数据集，是MOT17、CrowdHuman组成的联合数据集，**mix_det**数据集是MOT17、CrowdHuman、Cityscapes、ETHZ组成的联合数据集，数据集整理的格式和目录可以参考[此链接](https://github.com/ifzhang/ByteTrack#data-preparation)，最终放置于`dataset/mot/`目录下。为了验证精度可以都用**MOT17-half val**数据集去评估。


#### YOLOX-x ByteTrack(mix_det)在 MOT-16/MOT-17 上的结果

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pp-yoloe-an-evolved-version-of-yolo/multi-object-tracking-on-mot16)](https://paperswithcode.com/sota/multi-object-tracking-on-mot16?p=pp-yoloe-an-evolved-version-of-yolo)

|    网络      |  测试集 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :---------: | :-------: | :----: | :----: | :----: | :----: | :----: | :------: | :----: |:-----: |
| ByteTrack-x| MOT-17 Train |  84.4  |  72.8  |  837  |  5653  | 10985 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams) | [配置文件](./bytetrack_yolox.yml) |
| ByteTrack-x| **MOT-17 Test** |  **78.4**  |  69.7  |  4974  |  37551  | 79524 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams) | [配置文件](./bytetrack_yolox.yml) |
| ByteTrack-x| MOT-16 Train |  83.5  |  72.7  |  800  |  6973  | 10419 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams) | [配置文件](./bytetrack_yolox.yml) |
| ByteTrack-x| **MOT-16 Test** |  **77.7**  |  70.1  |  1570  |  15695  | 23304 |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams) | [配置文件](./bytetrack_yolox.yml) |


**注意:**
  - **mix_det**数据集是MOT17、CrowdHuman、Cityscapes、ETHZ组成的联合数据集，数据集整理的格式和目录可以参考[此链接](https://github.com/ifzhang/ByteTrack#data-preparation)，最终放置于`dataset/mot/`目录下。
  - MOT-17 Train 和 MOT-16 Train 的指标均为本地评估该数据后的指标，由于Train集包括在了训练集中，此MOTA指标不代表模型的检测跟踪能力，只是因为MOT-17和MOT-16无验证集而它们的Train集有ground truth，是为了方便验证精度。
  - MOT-17 Test 和 MOT-16 Test 的指标均为交到 [MOTChallenge](https://motchallenge.net)官网评测后的指标，因为MOT-17和MOT-16的Test集未开放ground truth，此MOTA指标可以代表模型的检测跟踪能力。
  - ByteTrack的训练是单独的检测器训练MOT数据集，推理是组装跟踪器去评估MOT指标，单独的检测模型也可以评估检测指标。
  - ByteTrack的导出部署，是单独导出检测模型，再组装跟踪器运行的，参照[PP-Tracking](../../../deploy/pptracking/python/README.md)。


### 人头跟踪

#### YOLOX-x ByteTrack 在 HT-21 Test Set上的结果

|    模型      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| ByteTrack-x     | 1440x800 |  64.1 |  63.4  |  4191   |  185162  |  210240 |    -     | [下载链接](https://paddledet.bj.bcebos.com/models/mot/bytetrack_yolox_ht21.pdparams) | [配置文件](./bytetrack_yolox_ht21.yml) |

#### YOLOX-x ByteTrack 在 HT-21 Test Set上的结果

|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| ByteTrack-x     | 1440x800 |  72.6  |  61.8  |  5163   |  71235  |  154139 |    -     | [下载链接](https://paddledet.bj.bcebos.com/models/mot/bytetrack_yolox_ht21.pdparams) | [配置文件](./bytetrack_yolox_ht21.yml) |

**注意:**
  - 更多人头跟踪模型可以参考[headtracking21](../headtracking21)。


## 多类别适配

多类别ByteTrack，可以参考 [bytetrack_ppyoloe_ppvehicle9cls.yml](./bytetrack_ppyoloe_ppvehicle9cls.yml)，表示使用 [PP-Vehicle](../../ppvehicle/) 中的PPVehicle9cls数据集训好的模型权重去做多类别车辆跟踪。由于没有跟踪的ground truth标签无法做评估，故只做跟踪预测，只需修改`TestMOTDataset`确保路径存在，且其中的`anno_path`表示指定在一个`label_list.txt`中记录具体类别，需要自己手写，一行表示一个种类，注意路径`anno_path`如果写错或找不到则将默认使用COCO数据集80类的类别。

如需**更换检测器权重，可改动其中的`det_weights: `为自己的权重路径**，并注意**数据集路径、`label_list.txt`和类别数**做出相应更改。

预测多类别车辆跟踪：
```bash
# 下载demo视频
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/bdd100k_demo.mp4

# 使用PPYOLOE 多类别车辆检测模型
CUDA_VISIBLE_DEVICES=1 python tools/infer_mot.py -c configs/mot/bytetrack/bytetrack_ppyoloe_ppvehicle9cls.yml --video_file=bdd100k_demo.mp4 --scaled=True --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
 - `--save_videos`表示保存可视化视频，同时会保存可视化的图片在`{output_dir}/mot_outputs/`中，`{output_dir}`可通过`--output_dir`设置，默认文件夹名为`output`。


## 快速开始

### 1. 训练
通过如下命令一键式启动训练和评估
```bash
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp
# 或者
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml --eval --amp
```

**注意:**
  - ` --eval`是边训练边验证精度；`--amp`是混合精度训练避免溢出，推荐使用paddlepaddle2.2.2版本。

### 2. 评估
#### 2.1 评估检测效果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_det.pdparams
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
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/bytetrack/bytetrack_yolox.yml --scaled=True
```
**注意:**
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE YOLOv3则为False，如果使用通用检测模型则为True, 默认值是False。
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置，默认文件夹名为`output`。

### 3. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 下载demo视频
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/mot17_demo.mp4

# 使用PPYOLOe行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/bytetrack/bytetrack_ppyoloe.yml --video_file=mot17_demo.mp4 --scaled=True --save_videos
# 或者使用YOLOX行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/bytetrack/bytetrack_yolox.yml --video_file=mot17_demo.mp4 --scaled=True --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
 - `--save_videos`表示保存可视化视频，同时会保存可视化的图片在`{output_dir}/mot_outputs/`中，`{output_dir}`可通过`--output_dir`设置，默认文件夹名为`output`。


### 4. 导出预测模型

Step 1：导出检测模型
```bash
# 导出PPYOLOe行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
# 或者导出YOLOX行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/yolox_x_24e_800x1440_mix_det.pdparams
```

Step 2：导出ReID模型(可选步骤，默认不需要)
```bash
# 导出PPLCNet ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pplcnet.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/ppyoloe_crn_l_36e_640x640_mot17half/ --tracker_config=deploy/pptracking/python/tracker_config.yml --video_file=mot17_demo.mp4 --device=GPU --save_mot_txts
# 或者
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/yolox_x_24e_800x1440_mix_det/ --tracker_config=deploy/pptracking/python/tracker_config.yml --video_file=mot17_demo.mp4 --device=GPU --save_mot_txts
```

**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)或`--save_mot_txt_per_img`(对每张图片保存一个txt)表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。


## 引用
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
