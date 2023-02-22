[English](PPVehicle_QUICK_STARTED_en.md) | 简体中文

# PP-Vehicle快速开始

## 目录

- [环境准备](#环境准备)
- [模型下载](#模型下载)
- [配置文件说明](#配置文件说明)
- [预测部署](#预测部署)
  - [在线视频流](#在线视频流)
  - [Jetson部署说明](#Jetson部署说明)
  - [参数说明](#参数说明)
- [方案介绍](#方案介绍)
  - [车辆检测](#车辆检测)
  - [车辆跟踪](#车辆跟踪)
  - [车牌识别](#车牌识别)
  - [属性识别](#属性识别)
  - [违章停车识别](#违章停车识别)


## 环境准备

环境要求： PaddleDetection版本 >= release/2.5 或 develop版本

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

PP-Vehicle提供了目标检测、属性识别、行为识别、ReID预训练模型，以实现不同使用场景，用户可以直接下载使用

| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  车辆检测（高精度）  | 25.7ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) | 182M |  
|  车辆检测（轻量级）  | 13.2ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) | 27M |
|  车辆检测（超轻量级）  | 10ms（Jetson AGX）  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz) | 17M |
|  车辆跟踪（高精度）  | 40ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) | 182M |
|  车辆跟踪（轻量级）  | 25ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) | 27M |
|  车辆跟踪（超轻量级）  | 13.2ms（Jetson AGX）  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz) | 17M |
|  车牌识别  |   4.68ms |  [车牌检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) <br> [车牌字符识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | 车牌检测：3.9M  <br> 车牌字符识别： 12M |
|  车辆属性  |   7.31ms | [车辆属性](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip) | 7.2M |
|  车道线检测  |   47ms | [车道线模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) | 47M |

下载模型后，解压至`./output_inference`文件夹。

在配置文件中，模型路径默认为模型的下载路径，如果用户不修改，则在推理时会自动下载对应的模型。

**注意：**

- 检测跟踪模型精度为公开数据集BDD100K-MOT和UA-DETRAC整合后的联合数据集PPVehicle的结果，具体参照[ppvehicle](../../../../configs/ppvehicle)
- 预测速度为T4下，开启TensorRT FP16的效果, 模型预测速度包含数据预处理、模型预测、后处理全流程

## 配置文件说明

PP-Vehicle相关配置位于```deploy/pipeline/config/infer_cfg_ppvehicle.yml```中，存放模型路径，完成不同功能需要设置不同的任务类型

功能及任务类型对应表单如下：

| 输入类型 | 功能 | 任务类型 | 配置项 |
|-------|-------|----------|-----|
| 图片 | 属性识别 | 目标检测 属性识别 | DET ATTR |
| 单镜头视频 | 属性识别 | 多目标跟踪 属性识别 | MOT ATTR |
| 单镜头视频 | 车牌识别 | 多目标跟踪 车牌识别 | MOT VEHICLEPLATE |

例如基于视频输入的属性识别，任务类型包含多目标跟踪和属性识别，具体配置如下：

```
crop_thresh: 0.5
visual: True
warmup_frame: 50

MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  enable: True

VEHICLE_ATTR:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip
  batch_size: 8
  color_threshold: 0.5
  type_threshold: 0.5
  enable: True
```

**注意：**

- 如果用户需要实现不同任务，可以在配置文件对应enable选项设置为True。
- 如果用户仅需要修改模型文件路径，可以在命令行中--config后面紧跟着 `-o MOT.model_dir=ppyoloe/` 进行修改即可，也可以手动修改配置文件中的相应模型路径，详细说明参考下方参数说明文档。


## 预测部署

1. 直接使用默认配置或者examples中配置文件，或者直接在`infer_cfg_ppvehicle.yml`中修改配置：
```
# 例：车辆检测，指定配置文件路径和测试图片
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml --image_file=test_image.jpg --device=gpu

# 例：车辆车牌识别，指定配置文件路径和测试视频
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_plate.yml --video_file=test_video.mp4 --device=gpu
```

2. 使用命令行进行功能开启，或者模型路径修改：
```
# 例：车辆跟踪，指定配置文件路径和测试视频，命令行中开启MOT模型并修改模型路径，命令行中指定的模型路径优先级高于配置文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml -o MOT.enable=True MOT.model_dir=ppyoloe_infer/ --video_file=test_video.mp4 --device=gpu

# 例：车辆违章分析，指定配置文件和测试视频，命令行中指定违停区域设置、违停时间判断。
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml \
                                                   --video_file=../car_test.mov \
                                                   --device=gpu \
                                                   --draw_center_traj \
                                                   --illegal_parking_time=3 \
                                                   --region_type=custom \
                                                   --region_polygon 600 300 1300 300 1300 800 600 800

```

### 在线视频流

在线视频流解码功能基于opencv的capture函数，支持rtsp、rtmp格式。

- rtsp拉流预测

对rtsp拉流的支持，使用--rtsp RTSP [RTSP ...]参数指定一路或者多路rtsp视频流，如果是多路地址中间用空格隔开。(或者video_file后面的视频地址直接更换为rtsp流地址)，示例如下：
```
# 例：车辆属性识别，单路视频流
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE]  --device=gpu

# 例：车辆属性识别，多路视频流
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE1]  rtsp://[YOUR_RTSP_SITE2] --device=gpu
```

- 视频结果推流rtsp

预测结果进行rtsp推流，使用--pushurl rtsp:[IP] 推流到IP地址端，PC端可以使用[VLC播放器](https://vlc.onl/)打开网络流进行播放，播放地址为 `rtsp:[IP]/videoname`。其中`videoname`是预测的视频文件名，如果视频来源是本地摄像头则`videoname`默认为`output`.
```
# 例：车辆属性识别，单路视频流，该示例播放地址为 rtsp://[YOUR_SERVER_IP]:8554/test_video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_attr.yml -o visual=False --video_file=test_video.mp4  --device=gpu --pushurl rtsp://[YOUR_SERVER_IP]:8554
```
注：
1. rtsp推流服务基于 [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server), 如使用推流功能请先开启该服务.
使用方法很简单，以linux平台为例：1）下载对应平台release包；2）解压后在命令行执行命令 `./rtsp-simple-server`即可，成功后进入服务开启状态就可以接收视频流了。
2. rtsp推流如果模型处理速度跟不上会出现很明显的卡顿现象，建议跟踪模型使用ppyoloe_s版本，即修改配置中跟踪模型mot_ppyoloe_l_36e_pipeline.zip替换为mot_ppyoloe_s_36e_pipeline.zip。

### Jetson部署说明

由于Jetson平台算力相比服务器有较大差距，有如下使用建议：

1. 模型选择轻量级版本，我们最新提供了轻量级[PP-YOLOE-Plus Tiny模型](../../../../configs/ppvehicle/README.md)，该模型在Jetson AGX上可以实现4路视频流20fps实时跟踪。
2. 如果需进一步提升速度，建议开启跟踪跳帧功能，推荐使用2或者3: `skip_frame_num: 3`，该功能当前默认关闭。

上述修改可以直接修改配置文件（推荐），也可以在命令行中修改（字段较长，不推荐）。

PP-YOLOE-Plus Tiny模型在AGX平台不同功能开启时的速度如下：（测试视频跟踪车辆为1个）

| 功能  | 平均每帧耗时(ms)  | 运行帧率(fps)  |
|:----------|:----------|:----------|
| 跟踪    | 13    | 77    |
| 属性识别    | 20.2    | 49.4    |
| 车牌识别    | -    | -    |


### 参数说明

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --config | Yes | 配置文件路径 |
| -o | Option | 覆盖配置文件中对应的配置  |
| --image_file | Option | 需要预测的图片 |
| --image_dir  | Option |  要预测的图片文件夹路径   |
| --video_file | Option | 需要预测的视频，或者rtsp流地址 |
| --rtsp | Option | rtsp视频流地址，支持一路或者多路同时输入 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --device | Option | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --pushurl | Option| 对预测结果视频进行推流的地址，以rtsp://开头，该选项优先级高于视频结果本地存储，打开时不再另外存储本地预测结果视频, 默认为空，表示没有开启|
| --output_dir | Option|可视化结果保存的根目录，默认为output/|
| --run_mode | Option |使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）|
| --enable_mkldnn | Option | CPU预测中是否开启MKLDNN加速，默认为False |
| --cpu_threads | Option| 设置cpu线程数，默认为1 |
| --trt_calib_mode | Option| TensorRT是否使用校准功能，默认为False。使用TensorRT的int8功能时，需设置为True，使用PaddleSlim量化后的模型时需要设置为False |
| --do_entrance_counting | Option | 是否统计出入口流量，默认为False |
| --draw_center_traj | Option | 是否绘制跟踪轨迹，默认为False |
| --region_type | Option | 'horizontal'（默认值）、'vertical'：表示流量统计方向选择；'custom'：表示设置车辆禁停区域 |
| --region_polygon | Option | 设置禁停区域多边形多点的坐标，无默认值 |
| --illegal_parking_time | Option | 设置禁停时间阈值，单位秒（s），-1（默认值）表示不做检查 |

## 方案介绍

PP-Vehicle 整体方案如下图所示:

<div width="1000" align="center">
  <img src="https://user-images.githubusercontent.com/31800336/218659932-31f4298c-042d-436d-9845-18879f5d31e3.png"/>
</div>


### 车辆检测
- 采用PP-YOLOE L 作为目标检测模型
- 详细文档参考[PP-YOLOE](../../../../configs/ppyoloe/)和[检测跟踪文档](ppvehicle_mot.md)

### 车辆跟踪
- 采用SDE方案完成车辆跟踪
- 检测模型使用PP-YOLOE L(高精度)和S(轻量级)
- 跟踪模块采用OC-SORT方案
- 详细文档参考[OC-SORT](../../../../configs/mot/ocsort)和[检测跟踪文档](ppvehicle_mot.md)

### 属性识别
- 使用PaddleClas提供的特色模型PP-LCNet，实现对车辆颜色及车型属性的识别。
- 详细文档参考[属性识别](ppvehicle_attribute.md)

### 车牌识别
- 使用PaddleOCR特色模型ch_PP-OCRv3_det+ch_PP-OCRv3_rec模型，识别车牌号码
- 详细文档参考[车牌识别](ppvehicle_plate.md)

### 违章停车识别
- 车辆跟踪模型使用高精度模型PP-YOLOE L，根据车辆的跟踪轨迹以及指定的违停区域判断是否违章停车，如果存在则展示违章停车车牌号。
- 详细文档参考[违章停车识别](ppvehicle_illegal_parking.md)

### 违法分析-逆行
- 违法分析-逆行，通过使用高精度分割模型PP-Seg，对车道线进行分割拟合，然后与车辆轨迹组合判断车辆行驶方向是否与道路方向一致。
- 详细文档参考[违法分析-逆行](ppvehicle_retrograde.md)

### 违法分析-压线
- 违法分析-逆行，通过使用高精度分割模型PP-Seg，对车道线进行分割拟合，然后与车辆区域是否覆盖实线区域，进行压线判断。
- 详细文档参考[违法分析-压线](ppvehicle_press.md)
