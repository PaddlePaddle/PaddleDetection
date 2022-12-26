简体中文 | [English](README.md)

# CenterTrack (Tracking Objects as Points)

## 内容
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 模型库

### MOT17

|      训练数据集     |  输入尺度  |  总batch_size  |  val MOTA  |  test MOTA  |     FPS   | 配置文件 |  下载链接|
| :---------------: | :-------: | :------------: | :--------: | :---------: | :-------: | :----: | :-----: |
| MOT17-half train |  544x960  |         32     |    -       |     -       |     -     |[config](./centertrack_dla34_70e_mot17half.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/centertrack_dla34_70e_mot17half.pdparams) |
| MOT17 train      |  544x960  |         32     |    -       |     -       |     -     |[config](./centertrack_dla34_70e_mot17.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/centertrack_dla34_70e_mot17.pdparams) |


**注意:**
  - **MOT17-half train**是MOT17的train序列(共7个)每个视频的前一半帧的图片和标注用作训练集，而用每个视频的后一半帧组成的**MOT17-half val**作为验证集去评估，数据集可以从[此链接](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip)下载，并解压放在`dataset/mot/`文件夹下。
  - **MOT17 train**是MOT17的train序列(共7个)用作训练集，同时也用**MOT17 train**数据集去评估，**test MOTA**为交到[MOT Challenge官网](https://motchallenge.net)评测的结果。
  - **mix_mot_ch**数据集，是MOT17、CrowdHuman组成的联合数据集，**mix_det**数据集是MOT17、CrowdHuman、Cityscapes、ETHZ组成的联合数据集，数据集整理的格式和目录可以参考[此链接](https://github.com/ifzhang/ByteTrack#data-preparation)，最终放置于`dataset/mot/`目录下。为了验证精度可以都用**MOT17-half val**数据集去评估。


## 快速开始

### 1.训练
通过如下命令一键式启动训练和评估
```bash
# 单卡训练(不推荐)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml --eval --amp
# 多卡训练
python -m paddle.distributed.launch --log_dir=centertrack_dla34_70e_mot17half/ --gpus 1,2,3,4,5,6,7 tools/train.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml --eval --amp
```
**注意:**
  - `--eval`是边训练边验证检测的mAP精度；`--amp`是混合精度训练避免溢出；


### 2.评估
#### 2.1 评估检测效果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml  -o weights=output/centertrack_dla34_70e_mot17half/model_final.pdparams
```

**注意:**
 - 评估检测使用的是```tools/eval.py```, 评估跟踪使用的是```tools/eval_mot.py```。

#### 2.2 评估跟踪效果

注意首先需要取消配置文件中的`mot_metric: True`和`metric: MOT`的注释:

```python
### for train.py/eval.py/infer.py
mot_metric: False
metric: COCO

### for eval_mot.py/infer_mot_mot.py
mot_metric: True # 默认是被注释的，评估跟踪需要为 True，会覆盖之前的 mot_metric: False
metric: MOT # 默认是被注释的，评估跟踪需要使用 MOT，会覆盖之前的 metric: COCO
```

然后执行以下语句：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml
```
**注意:**
 - 评估检测使用的是```tools/eval.py```, 评估跟踪使用的是```tools/eval_mot.py```。
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置，默认文件夹名为`output`。


### 3.预测

使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 下载demo视频
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/mot17_demo.mp4

CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml --video_file=mot17_demo.mp4 --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--save_videos`表示保存可视化视频，同时会保存可视化的图片在`{output_dir}/mot_outputs/`中，`{output_dir}`可通过`--output_dir`设置，默认文件夹名为`output`。


### 4. 导出预测模型

Step 1：导出检测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/centertrack/centertrack_dla34_70e_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/centertrack_dla34_70e_mot17half.pdparams
```

### 5. 用导出的模型基于Python去预测

```bash
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/centertrack_dla34_70e_mot17half/ --tracker_config=deploy/pptracking/python/tracker_config.yml --video_file=mot17_demo.mp4 --device=GPU --save_mot_txts
```

**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)或`--save_mot_txt_per_img`(对每张图片保存一个txt)表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。


## 引用
```
@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
}
```
