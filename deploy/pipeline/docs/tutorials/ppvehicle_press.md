[English](ppvehicle_press_en.md) | 简体中文

# PP-Vehicle压实线识别模块

车辆压实线识别在智慧城市，智慧交通等方向具有广泛应用。在PP-Vehicle中，集成了车辆压实线识别模块，可识别车辆是否违章压实线。

| 任务 | 算法 | 精度 | 预测速度 | 下载链接|
|-----------|------|-----------|----------|---------------|
| 车辆检测/跟踪 | PP-YOLOE | mAP 63.9 | 38.67ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| 车道线识别 | PP-liteseg | mIou 32.69 | 47 ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) |


注意：
1. 车辆检测/跟踪模型预测速度是基于NVIDIA T4, 开启TensorRT FP16得到。模型预测速度包含数据预处理、模型预测、后处理部分。
2. 车辆检测/跟踪模型的训练和精度测试均基于[VeRi数据集](https://www.v7labs.com/open-datasets/veri-dataset)。
3. 车道线模型预测速度基于Tesla P40,python端预测，模型预测速度包含数据预处理、模型预测、后处理部分。
4. 车道线模型训练和精度测试均基于[BDD100K-LaneSeg](https://bdd-data.berkeley.edu/portal.html#download)和[Apollo Scape](http://apolloscape.auto/lane_segmentation.html#to_dataset_href),两个数据集车道线分割[标签](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)


## 使用方法

### 配置项说明

[配置文件](../../config/infer_cfg_ppvehicle.yml)中与车辆压线相关的参数如下：
```
VEHICLE_PRESSING:
  enable: True               #是否开启功能
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml #车道线提取配置文件
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip   #模型文件路径
```
[车道线配置文件](../../config/lane_seg_config.yml)中与车道线提取相关的参数如下：
```
type: PLSLaneseg  #选择分割模型

PLSLaneseg:
  batch_size: 1                                       #图片batch_size
  device: gpu                                         #选择gpu还是cpu
  filter_flag: True                                   #是否过滤水平方向道路线
  horizontal_filtration_degree: 23                    #过滤水平方向车道线阈值，当分割出来的车道线最大倾斜角与
                                                      #最小倾斜角差值小于阈值时，不进行过滤
  horizontal_filtering_threshold: 0.25                #确定竖直方向与水平方向分开阈值
                                                      #thr = (min_degree+max_degree)*0.25
                                                      #根据车道线倾斜角与thr的大小比较，将车道线分为垂直方向与水平方向
```

### 使用命令

1. 从模型库下载`车辆检测/跟踪`, `车道线识别`两个预测部署模型并解压到`./output_inference`路径下；默认会自动下载模型，如果手动下载，需要修改模型文件夹为模型存放路径。
2. 修改配置文件中`VEHICLE_PRESSING`项的`enable: True`，以启用该功能。
3. 图片输入时，启动命令如下(更多命令参数说明，请参考[快速开始-参数说明](./PPVehicle_QUICK_STARTED.md))：

```bash
# 预测单张图片文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --image_file=test_image.jpg \
                                   --device=gpu

# 预测包含一张或多张图片的文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --image_dir=images/ \
                                   --device=gpu
```

4. 视频输入时，启动命令如下：

```bash
#预测单个视频文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --video_file=test_video.mp4 \
                                   --device=gpu

#预测包含一个或多个视频的文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   --video_dir=test_videos/ \
                                   -o VEHICLE_PRESSING.enable=true
                                   --device=gpu
```

5. 若修改模型路径，有以下两种方式：

    - 方法一：`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`下可以配置不同模型路径，车道线识别模型修改`LANE_SEG`字段下配置
    - 方法二：直接在命令行中增加`-o`，以覆盖配置文件中的默认模型路径：

```bash
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   -o VEHICLE_PRESSING.enable=true
                                                   LANE_SEG.model_dir=output_inference
```

测试效果如下：

<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_press.gif"/>
</div>

## 方案说明
1.车道线识别模型使用了[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 的超轻量分割方案。训练样本[标签](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)分为4类：
  0 背景
  1 双黄线
  2 实线
  3 虚线
车辆压线分析过滤虚线类；

2.车道线通过对分割结果聚类得到，且默认过滤水平方向车道线，若不过滤可在[车道线配置文件](../../config/lane_seg_config.yml)修改`filter_flag`参数；

3.车辆压线判断条件：车辆的检测框底边线与车道线是否有交点；

**性能优化措施**
1.因摄像头视角原因，可以根据实际情况决定是否过滤水平方向车道线;
