# 基于视频分类的行为识别

## 数据准备

视频分类任务输入的视频格式一般为`.mp4`、`.avi`等格式视频或者是抽帧后的视频帧序列，标签则可以是`.txt`格式存储的文件。

对于打架识别任务，具体数据准备流程如下：

### 数据集下载

打架识别基于6个公开的打架、暴力行为相关数据集合并后的数据进行模型训练。公开数据集具体信息如下：

| 数据集 | 下载连接 | 简介 | 标注 | 数量 | 时长 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Surveillance Camera Fight Dataset| https://github.com/sayibet/fight-detection-surv-dataset | 裁剪视频，监控视角 | 视频级别 | 打架：150；非打架：150 | 2s |
| A Dataset for Automatic Violence Detection in Videos | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | 裁剪视频，室内自行录制 | 视频级别 | 暴力行为：115个场景，2个机位，共230 ；非暴力行为：60个场景，2个机位，共120 | 几秒钟 |
| Hockey Fight Detection Dataset | https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes?resource=download | 裁剪视频，非真实场景 | 视频级别 | 打架：500；非打架：500 | 2s |
| Video Fight Detection Dataset | https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset | 裁剪视频，非真实场景 | 视频级别 | 打架：100；非打架：101 | 2s |
| Real Life Violence Situations Dataset | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | 裁剪视频，非真实场景 | 视频级别 | 暴力行为：1000；非暴力行为：1000 | 几秒钟 |
| UBI Abnormal Event Detection Dataset| http://socia-lab.di.ubi.pt/EventDetection/ | 未裁剪视频，监控视角 | 帧级别 | 打架：216；非打架：784；裁剪后二次标注：打架1976，非打架1630 | 原视频几秒到几分钟不等，裁剪后2s |

打架（暴力行为）视频3956个，非打架（非暴力行为）视频3501个，共7457个视频，每个视频几秒钟。

本项目为大家整理了前5个数据集，下载链接：[https://aistudio.baidu.com/aistudio/datasetdetail/149085](https://aistudio.baidu.com/aistudio/datasetdetail/149085)。

### 视频抽帧

首先下载PaddleVideo代码：
```bash
git clone https://github.com/PaddlePaddle/PaddleVideo.git
```

假设PaddleVideo源码路径为PaddleVideo_root。

为了加快训练速度，将视频进行抽帧。下面命令会根据视频的帧率FPS进行抽帧，如FPS=30，则每秒视频会抽取30帧图像。

```bash
cd ${PaddleVideo_root}
python data/ucf101/extract_rawframes.py dataset/ rawframes/ --level 2 --ext mp4
```
其中，假设视频已经存放在了`dataset`目录下，如果是其他路径请对应修改。打架（暴力）视频存放在`dataset/fight`中；非打架（非暴力）视频存放在`dataset/nofight`中。`rawframes`目录存放抽取的视频帧。

### 训练集和验证集划分

打架识别验证集1500条，来自Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、UBI Abnormal Event Detection Dataset三个数据集。

也可根据下面的命令将数据按照8:2的比例划分成训练集和测试集：

```bash
python split_fight_train_test_dataset.py "rawframes" 2 0.8
```

参数说明：“rawframes”为视频帧存放的文件夹；2表示目录结构为两级，第二级表示每个行为对应的子文件夹；0.8表示训练集比例。

其中`split_fight_train_test_dataset.py`文件在PaddleDetection中的`deploy/pipeline/tools`路径下。

执行完命令后会最终生成fight_train_list.txt和fight_val_list.txt两个文件。打架的标签为1，非打架的标签为0。

### 视频裁剪
对于未裁剪的视频，如UBI Abnormal Event Detection Dataset数据集，需要先进行裁剪才能用于模型训练，`deploy/pipeline/tools/clip_video.py`中给出了视频裁剪的函数`cut_video`，输入为视频路径，裁剪的起始帧和结束帧以及裁剪后的视频保存路径。


## 模型优化

### VideoMix
[VideoMix](https://arxiv.org/abs/2012.03457)是视频数据增强的方法之一，是对图像数据增强CutMix的扩展，可以缓解模型的过拟合问题。

与Mixup将两个视频片段的每个像素点按照一定比例融合不同的是，VideoMix是每个像素点要么属于片段A要么属于片段B。输出结果是两个片段原始标签的加权和，权重是两个片段各自的比例。

在baseline的基础上加入VideoMix数据增强后，精度由87.53%提升至88.01%。

### 更大的分辨率
由于监控摄像头角度、距离等问题，存在监控画面下人比较小的情况，小目标行为的识别较困难，尝试增大输入图像的分辨率，模型精度由88.01%提升至89.06%。

## 新增行为

目前打架识别模型使用的是[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)套件中[PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)，并在PP-TSM视频分类模型训练流程的基础上修改适配，完成模型训练。

请先参考[使用说明](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/usage.md)了解PaddleVideo模型库的使用。


| 任务 | 算法 | 精度 | 预测速度(ms) | 模型权重 | 预测部署模型 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  打架识别 | PP-TSM | 准确率：89.06% | T4, 2s视频128ms | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

#### 模型训练
下载预训练模型：
```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

执行训练：
```bash
# 单卡训练
cd ${PaddleVideo_root}
python main.py --validate -c pptsm_fight_frames_dense.yaml
```

本方案针对的是视频的二分类问题，如果不是二分类，需要修改配置文件中`MODEL-->head-->num_classes`为具体的类别数目。


```bash
cd ${PaddleVideo_root}
# 多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -B -m paddle.distributed.launch --gpus=“0,1,2,3” \
   --log_dir=log_pptsm_dense  main.py  --validate \
   -c pptsm_fight_frames_dense.yaml
```

#### 模型评估
训练好的模型下载：[https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams)

模型评估：
```bash
cd ${PaddleVideo_root}
python main.py --test -c pptsm_fight_frames_dense.yaml \
   -w ppTSM_fight_best.pdparams
```

其中`ppTSM_fight_best.pdparams`为训练好的模型。

#### 模型导出

导出inference模型：

```bash
cd ${PaddleVideo_root}
python tools/export_model.py -c pptsm_fight_frames_dense.yaml \
                                -p ppTSM_fight_best.pdparams \
                                -o inference/ppTSM
```


#### 推理可视化

利用上步骤导出的模型，基于PaddleDetection中推理pipeline可完成自定义行为识别及可视化。

新增行为后，需要对现有的可视化代码进行修改，目前代码支持打架二分类可视化，新增类别后需要根据识别结果自适应可视化推理结果。

具体修改PaddleDetection中/deploy/pipeline/pipeline.py路径下PipePredictor类中visualize_video成员函数。当结果中存在'video_action'数据时，会对行为进行可视化。目前的逻辑是如果推理的类别为1，则为打架行为，进行可视化；否则不进行显示，即"video_action_score"为None。用户新增行为后，可根据类别index和对应的行为设置"video_action_text"字段，目前index=1对应"Fight"。相关代码块如下：

```
video_action_res = result.get('video_action')
if video_action_res is not None:
   video_action_score = None
   if video_action_res and video_action_res["class"] == 1:
         video_action_score = video_action_res["score"]
   mot_boxes = None
   if mot_res:
         mot_boxes = mot_res['boxes']
   image = visualize_action(
         image,
         mot_boxes,
         action_visual_collector=None,
         action_text="SkeletonAction",
         video_action_score=video_action_score,
         video_action_text="Fight")
```
