[English](PPHuman_QUICK_STARTED_en.md) | 简体中文

# PP-Human快速开始

## 目录

- [环境准备](#环境准备)
- [模型下载](#模型下载)
- [配置文件说明](#配置文件说明)
- [预测部署](#预测部署)
  - [在线视频流](#在线视频流)
  - [Jetson部署说明](#Jetson部署说明)
  - [参数说明](#参数说明)
- [方案介绍](#方案介绍)
  - [行人检测](#行人检测)
  - [行人跟踪](#行人跟踪)
  - [跨镜行人跟踪](#跨镜行人跟踪)
  - [属性识别](#属性识别)
  - [行为识别](#行为识别)

## 环境准备

环境要求： PaddleDetection版本 >= release/2.4 或 develop版本

PaddlePaddle和PaddleDetection安装

```
# PaddlePaddle CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# PaddlePaddle CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# 克隆PaddleDetection仓库
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# 安装其他依赖
cd PaddleDetection
pip install -r requirements.txt
```

1. 详细安装文档参考[文档](../../../../docs/tutorials/INSTALL_cn.md)
2. 如果需要TensorRT推理加速（测速方式），请安装带`TensorRT版本Paddle`。您可以从[Paddle安装包](https://paddleinference.paddlepaddle.org.cn/v2.2/user_guides/download_lib.html#python)下载安装，或者按照[指导文档](https://www.paddlepaddle.org.cn/inference/master/optimize/paddle_trt.html)使用docker或自编译方式准备Paddle环境。

## 模型下载

PP-Human提供了目标检测、属性识别、行为识别、ReID预训练模型，以实现不同使用场景，用户可以直接下载使用

| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  行人检测（高精度）  | 25.1ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人检测（轻量级）  | 16.2ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
| 行人检测（超轻量级） | 10ms(Jetson AGX)   | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)        | 17M  |
|  行人跟踪（高精度）  | 31.8ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |
|  行人跟踪（轻量级）  | 21.0ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
| 行人跟踪（超轻量级） | 13.2ms(Jetson AGX)    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)        | 17M  |
|  跨镜跟踪(REID)   |   单人1.5ms | [REID](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) | REID：92M |
|  属性识别（高精度）  |   单人8.5ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  属性识别（轻量级）  |   单人7.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  摔倒识别  |   单人10ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
|  闯入识别  |   31.8ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 多目标跟踪：182M |
|  打架识别  |   19.7ms | [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 90M |
|  抽烟识别  |   单人15.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip) | 目标检测：182M<br>基于人体id的目标检测：27M |
|  打电话识别  |   单人6.0ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) | 目标检测：182M<br>基于人体id的图像分类：45M |

下载模型后，解压至`./output_inference`文件夹。

在配置文件中，模型路径默认为模型的下载路径，如果用户不修改，则在推理时会自动下载对应的模型。

**注意：**

- 模型精度为融合数据集结果，数据集包含开源数据集和企业数据集
- ReID模型精度为Market1501数据集测试结果
- 预测速度为T4下，开启TensorRT FP16的效果, 模型预测速度包含数据预处理、模型预测、后处理全流程

## 配置文件说明

PP-Human相关配置位于```deploy/pipeline/config/infer_cfg_pphuman.yml```中，存放模型路径，该配置文件中包含了目前PP-Human支持的所有功能。如果想要查看某个单一功能的配置，请参见```deploy/pipeline/config/examples/```中相关配置。此外，配置文件中的内容可以通过```-o```命令行参数修改，如修改属性的模型目录，则可通过```-o ATTR.model_dir="DIR_PATH"```进行设置。

功能及任务类型对应表单如下：

| 输入类型 | 功能 | 任务类型 | 配置项 |
|-------|-------|----------|-----|
| 图片 | 属性识别 | 目标检测 属性识别 | DET ATTR |
| 单镜头视频 | 属性识别 | 多目标跟踪 属性识别 | MOT ATTR |
| 单镜头视频 | 行为识别 | 多目标跟踪 关键点检测 摔倒识别 | MOT KPT SKELETON_ACTION |

例如基于视频输入的属性识别，任务类型包含多目标跟踪和属性识别，具体配置如下：

```
crop_thresh: 0.5
attr_thresh: 0.5
visual: True

MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  enable: True

ATTR:
  model_dir:  https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip
  batch_size: 8
  enable: True
```

**注意：**

- 如果用户需要实现不同任务，可以在配置文件对应enable选项设置为True。


## 预测部署

1. 直接使用默认配置或者examples中配置文件，或者直接在`infer_cfg_pphuman.yml`中修改配置：
```
# 例：行人检测，指定配置文件路径和测试图片，图片输入默认打开检测模型
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=gpu

# 例：行人属性识别，直接使用examples中配置
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml --video_file=test_video.mp4 --device=gpu
```

2. 使用命令行进行功能开启，或者模型路径修改：
```
# 例：行人跟踪，指定配置文件路径，模型路径和测试视频, 命令行中指定的模型路径优先级高于配置文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml -o MOT.enable=True MOT.model_dir=ppyoloe_infer/ --video_file=test_video.mp4 --device=gpu

# 例：行为识别，以摔倒识别为例，命令行中开启SKELETON_ACTION模型
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml -o SKELETON_ACTION.enbale=True --video_file=test_video.mp4 --device=gpu
```

### 在线视频流

在线视频流解码功能基于opencv的capture函数，支持rtsp、rtmp格式。

- rtsp拉流预测

对rtsp拉流的支持，使用--rtsp RTSP [RTSP ...]参数指定一路或者多路rtsp视频流，如果是多路地址中间用空格隔开。(或者video_file后面的视频地址直接更换为rtsp流地址)，示例如下：
```
# 例：行人属性识别，单路视频流
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE]  --device=gpu

# 例：行人属性识别，多路视频流
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE1]  rtsp://[YOUR_RTSP_SITE2] --device=gpu
```

- 视频结果推流rtsp

预测结果进行rtsp推流，使用--pushurl rtsp:[IP] 推流到IP地址端，PC端可以使用[VLC播放器](https://vlc.onl/)打开网络流进行播放，播放地址为 `rtsp:[IP]/videoname`。其中`videoname`是预测的视频文件名，如果视频来源是本地摄像头则`videoname`默认为`output`.
```
# 例：行人属性识别，单路视频流，该示例播放地址为 rtsp://[YOUR_SERVER_IP]:8554/test_video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml --video_file=test_video.mp4 --device=gpu --pushurl rtsp://[YOUR_SERVER_IP]:8554
```
注：
1. rtsp推流服务基于 [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server), 如使用推流功能请先开启该服务.
使用方法很简单，以linux平台为例：1）下载对应平台release包；2）解压后在命令行执行命令 `./rtsp-simple-server`即可，成功后进入服务开启状态就可以接收视频流了。
2. rtsp推流如果模型处理速度跟不上会出现很明显的卡顿现象，建议跟踪模型使用ppyoloe_s或ppyoloe-plus-tiny版本，方式为修改配置中跟踪模型mot_ppyoloe_l_36e_pipeline.zip替换为mot_ppyoloe_s_36e_pipeline.zip。


### Jetson部署说明

由于Jetson平台算力相比服务器有较大差距，有如下使用建议：

1. 模型选择轻量级版本，我们最新提供了轻量级[PP-YOLOE-Plus Tiny模型](../../../../configs/pphuman/README.md)，该模型在Jetson AGX上可以实现4路视频流20fps实时跟踪。
2. 如果需进一步提升速度，建议开启跟踪跳帧功能，推荐使用2或者3: `skip_frame_num: 3`，该功能当前默认关闭。

上述修改可以直接修改配置文件（推荐），也可以在命令行中修改（字段较长，不推荐）。

PP-YOLOE-Plus Tiny模型在AGX平台不同功能开启时的速度如下：（跟踪人数为3人情况下，以属性为例，总耗时为跟踪13.3+5.2*3≈29ms）

| 功能  | 平均每帧耗时(ms)  | 运行帧率(fps)  |
|:----------|:----------|:----------|
| 跟踪    | 13    | 77    |
| 属性识别    | 29    | 34    |
| 摔倒识别    | 64.5    | 15.5    |
| 抽烟识别    | 68.8    | 14.5    |
| 打电话识别    | 22.5    | 44.5    |
| 打架识别    | 3.98    | 251    |



### 参数说明

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --config | Yes | 配置文件路径 |
| -o | Option | 覆盖配置文件中对应的配置  |
| --image_file | Option | 需要预测的图片 |
| --image_dir  | Option |  要预测的图片文件夹路径   |
| --video_file | Option | 需要预测的视频，或者rtsp流地址（推荐使用rtsp参数） |
| --rtsp | Option | rtsp视频流地址，支持一路或者多路同时输入 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --device | Option | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --pushurl | Option| 对预测结果视频进行推流的地址，以rtsp://开头，该选项优先级高于视频结果本地存储，打开时不再另外存储本地预测结果视频|
| --output_dir | Option|可视化结果保存的根目录，默认为output/|
| --run_mode | Option |使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）|
| --enable_mkldnn | Option | CPU预测中是否开启MKLDNN加速，默认为False |
| --cpu_threads | Option| 设置cpu线程数，默认为1 |
| --trt_calib_mode | Option| TensorRT是否使用校准功能，默认为False。使用TensorRT的int8功能时，需设置为True，使用PaddleSlim量化后的模型时需要设置为False |
| --do_entrance_counting | Option | 是否统计出入口流量，默认为False |
| --draw_center_traj | Option | 是否绘制跟踪轨迹，默认为False |
| --region_type | Option | 'horizontal'（默认值）、'vertical'：表示流量统计方向选择；'custom'：表示设置闯入区域 |
| --region_polygon | Option | 设置闯入区域多边形多点的坐标，无默认值 |
| --do_break_in_counting | Option | 此项表示做区域闯入检查 |

## 方案介绍

PP-Human v2整体方案如下图所示:

<div width="1000" align="center">
  <img src="../../../../docs/images/pphumanv2.png"/>
</div>


### 行人检测
- 采用PP-YOLOE L 作为目标检测模型
- 详细文档参考[PP-YOLOE](../../../../configs/ppyoloe/)和[检测跟踪文档](pphuman_mot.md)

### 行人跟踪
- 采用SDE方案完成行人跟踪
- 检测模型使用PP-YOLOE L(高精度)和S(轻量级)
- 跟踪模块采用OC-SORT方案
- 详细文档参考[OC-SORT](../../../../configs/mot/ocsort)和[检测跟踪文档](pphuman_mot.md)

### 跨镜行人跟踪
- 使用PP-YOLOE + OC-SORT得到单镜头多目标跟踪轨迹
- 使用ReID（StrongBaseline网络）对每一帧的检测结果提取特征
- 多镜头轨迹特征进行匹配，得到跨镜头跟踪结果
- 详细文档参考[跨镜跟踪](pphuman_mtmct.md)

### 属性识别
- 使用PP-YOLOE + OC-SORT跟踪人体
- 使用PP-HGNet、PP-LCNet（多分类模型）完成识别属性，主要属性包括年龄、性别、帽子、眼睛、上衣下衣款式、背包等
- 详细文档参考[属性识别](pphuman_attribute.md)

### 行为识别：
- 提供四种行为识别方案
- 1. 基于骨骼点的行为识别，例如摔倒识别
- 2. 基于图像分类的行为识别，例如打电话识别
- 3. 基于检测的行为识别，例如吸烟识别
- 4. 基于视频分类的行为识别，例如打架识别
- 详细文档参考[行为识别](pphuman_action.md)
