[English](fight_recognition_en.md) | 简体中文

# 打架识别模型

## 内容
- [1 快速开始](#快速开始)
- [2 数据准备](#数据准备)
    - [2.1 数据集下载](#数据集下载)
    - [2.2 视频抽帧](#视频抽帧)
    - [2.3 训练集和验证集划分](#训练集和验证集划分)
    - [2.4 视频裁剪](#视频裁剪)
- [3 模型训练](#模型训练)
- [4 模型评估](#模型评估)
- [5 模型导出](#模型导出)


实时行人分析工具[PP-Human](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman)中集成了视频分类的打架识别模块。本文档介绍如何基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/)，完成打架识别模型的训练流程。

目前打架识别模型使用的是[PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/63c88a435e98c6fcaf353429d2df6cc24b8113ba/docs/zh-CN/model_zoo/recognition/pp-tsm.md)，并在PP-TSM视频分类模型训练流程的基础上修改适配，完成模型训练。

请先参考[使用说明](https://github.com/XYZ-916/PaddleVideo/blob/develop/docs/zh-CN/usage.md)了解PaddleVideo模型库的使用。


| 任务 | 算法 | 精度 | 预测速度(ms) | 模型权重 | 预测部署模型 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  打架识别 | PP-TSM | 准确率：89.06% | T4, 2s视频128ms | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

<a name="快速开始"></a>
## 1 快速开始

打架识别静态图模型获取[https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip)。

打架识别[demo](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/fight_demo.mp4)。

首先需要将下载好的静态图模型解压并放到`inference`目录下，然后执行下面的命令即可直接判断一个给定的视频中是否存在打架行为：

```
cd ${PaddleVideo_root}
python tools/predict.py --input_file fight.avi \
                           --config pptsm_fight_frames_dense.yaml \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


<a name="数据准备"></a>
## 2 数据准备

PP-TSM是一个基于视频片段进行预测的模型。在PaddleVideo中，训练数据为`.mp4`、`.avi`等格式视频或者是抽帧后的视频帧序列，标签则可以是`.txt`格式存储的文件。

<a name="数据集下载"></a>
### 2.1 数据集下载

本项目基于6个公开的打架、暴力行为相关数据集合并后的数据进行模型训练。公开数据集具体信息如下：

| 数据集 | 下载连接 | 简介 | 标注 | 数量 | 时长 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Surveillance Camera Fight Dataset| https://github.com/sayibet/fight-detection-surv-dataset | 裁剪视频，监控视角 | 视频级别 | 打架：150；非打架：150 | 2s |
| A Dataset for Automatic Violence Detection in Videos | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | 裁剪视频，室内自行录制 | 视频级别 | 暴力行为：115个场景，2个机位，共230 ；非暴力行为：60个场景，2个机位，共120 | 几秒钟 |
| Hockey Fight Detection Dataset | https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes?resource=download | 裁剪视频，非真实场景 | 视频级别 | 打架：500；非打架：500 | 2s |
| Video Fight Detection Dataset | https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset | 裁剪视频，非真实场景 | 视频级别 | 打架：100；非打架：101 | 2s |
| Real Life Violence Situations Dataset | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | 裁剪视频，非真实场景 | 视频级别 | 暴力行为：1000；非暴力行为：1000 | 几秒钟 |
| UBI Abnormal Event Detection Dataset| http://socia-lab.di.ubi.pt/EventDetection/ | 未裁剪视频，监控视角 | 帧级别 | 打架：216；非打架：784；裁剪后二次标注：打架1976，非打架1630 | 原视频几秒到几分钟不等，裁剪后2s |

打架（暴力行为）视频3956个，非打架（非暴力行为）视频3501个，共7457个视频，每个视频几秒钟。

<a name="视频抽帧"></a>
### 2.2 视频抽帧

为了加快训练速度，将视频进行抽帧。下面命令会根据视频的帧率FPS进行抽帧，如FPS=30，则每秒视频会抽取30帧图像。

```bash
cd ${PaddleVideo_root}
python data/ucf101/extract_rawframes.py dataset/ rawframes/ --level 2 --ext mp4
```
其中，视频存放在`dataset`目录下，打架（暴力）视频存放在`dataset/fight`中；非打架（非暴力）视频存放在`dataset/nofight`中。`rawframes`目录存放抽取的视频帧。

<a name="训练集和验证集划分"></a>
### 2.3 训练集和验证集划分

本项目验证集1500条，来自Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、UBI Abnormal Event Detection Dataset三个数据集。

也可根据下面的代码将数据按照8:2的比例划分成训练集和测试集：

```python
import os
import glob
import random
import fnmatch
import re

class_id = {
    "nofight":0,
    "fight":1
}

def get_list(path,key_func=lambda x: x[-11:], rgb_prefix='img_', level=1):
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory):
        lst = os.listdir(directory)
        cnt = len(fnmatch.filter(lst, rgb_prefix + '*'))
        return cnt

    # check RGB
    video_dict = {}
    for f in frame_folders:
        cnt = count_files(f)
        k = key_func(f)
        if level==2:
            k = k.split("/")[0]

        video_dict[f]=str(cnt)+" "+str(class_id[k])

    return video_dict

def fight_splits(video_dict, train_percent=0.8):
    videos = list(video_dict.keys())

    train_num = int(len(videos)*train_percent)

    train_list = []
    val_list = []

    random.shuffle(videos)

    for i in range(train_num):
        train_list.append(videos[i]+" "+str(video_dict[videos[i]]))
    for i in range(train_num,len(videos)):
        val_list.append(videos[i]+" "+str(video_dict[videos[i]]))

    print("train:",len(train_list),",val:",len(val_list))

    with open("fight_train_list.txt","w") as f:
        for item in train_list:
            f.write(item+"\n")

    with open("fight_val_list.txt","w") as f:
        for item in val_list:
            f.write(item+"\n")

frame_dir = "rawframes"
level = 2
train_percent = 0.8

if level == 2:
    def key_func(x):
        return '/'.join(x.split('/')[-2:])
else:
    def key_func(x):
        return x.split('/')[-1]

video_dict = get_list(frame_dir, key_func=key_func, level=level)  
print("number:",len(video_dict))

fight_splits(video_dict, train_percent)
```

最终生成fight_train_list.txt和fight_val_list.txt两个文件。打架的标签为1，非打架的标签为0。

<a name="视频裁剪"></a>
### 2.4 视频裁剪
对于未裁剪的视频，需要先进行裁剪才能用于模型训练，这个给出视频裁剪的函数`cut_video`，输入为视频路径，裁剪的起始帧和结束帧以及裁剪后的视频保存路径。

```python

import cv2

def cut_video(video_path, frameToStart, frametoStop, saved_video_path):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoWriter =cv2.VideoWriter(saved_video_path,apiPreference = 0,fourcc = cv2.VideoWriter_fourcc(*'mp4v'),fps=FPS,
            frameSize=(int(size[0]),int(size[1])))

    COUNT = 0
    while True:
            success, frame = cap.read()
            if success:
                COUNT += 1
                if COUNT <= frametoStop and COUNT > frameToStart:  # 选取起始帧
                    videoWriter.write(frame)
            else:
                print("cap.read failed!")
                break
            if COUNT > frametoStop:
                break

    cap.release()
    videoWriter.release()

    print(saved_video_path)
```

<a name="模型训练"></a>
## 3 模型训练
下载预训练模型：
```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

模型训练：
```bash
# 单卡训练
cd ${PaddleVideo_root}
python main.py --validate -c pptsm_fight_frames_dense.yaml
```

```bash
cd ${PaddleVideo_root}
# 多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -B -m paddle.distributed.launch --gpus=“0,1,2,3” \
   --log_dir=log_pptsm_dense  main.py  --validate \
   -c pptsm_fight_frames_dense.yaml
```

<a name="模型评估"></a>
## 4 模型评估

训练好的模型下载：[https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams)

模型评估：
```bash
cd ${PaddleVideo_root}
python main.py --test -c pptsm_fight_frames_dense.yaml \
   -w ppTSM_fight_best.pdparams
```

其中`ppTSM_fight_best.pdparams`为训练好的模型。

<a name="模型导出"></a>
## 5 模型导出

导出inference模型：

```bash
cd ${PaddleVideo_root}
python tools/export_model.py -c pptsm_fight_frames_dense.yaml \
                                -p ppTSM_fight_best.pdparams \
                                -o inference/ppTSM
```
