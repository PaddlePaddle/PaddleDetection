[English](ppvehicle_retrograde_en.md) | 简体中文

# PP-Vehicle车辆逆行识别模块

车辆逆行识别在智慧城市，智慧交通等方向具有广泛应用。在PP-Vehicle中，集成了车辆逆行识别模块，可识别车辆是否逆行。

| 任务 | 算法 | 精度 | 预测速度 | 下载链接|
|-----------|------|-----------|----------|---------------|
| 车辆检测/跟踪 | PP-YOLOE | mAP 63.9 | 38.67ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| 车道线识别 | PP-liteseg | mIou 32.69 | 47 ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) |


注意：
1. 车辆检测/跟踪模型预测速度是基于NVIDIA T4, 开启TensorRT FP16得到。模型预测速度包含数据预处理、模型预测、后处理部分。
2. 车辆检测/跟踪模型的训练和精度测试均基于[VeRi数据集](https://www.v7labs.com/open-datasets/veri-dataset)。
3. 车道线模型预测速度基于Tesla P40,python端预测，模型预测速度包含数据预处理、模型预测、后处理部分。
4. 车道线模型训练和精度测试均基于[BDD100K-LaneSeg](https://bdd-data.berkeley.edu/portal.html#download.)和[Apollo Scape](http://apolloscape.auto/lane_segmentation.html#to_dataset_href)。两个数据集的标签文件[Lane_dataset_label](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)


## 使用方法

### 配置项说明

[配置文件](../../config/infer_cfg_ppvehicle.yml)中与车辆逆行识别模块相关的参数如下：
```
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml #车道线提取配置文件
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip   #模型文件路径

VEHICLE_RETROGRADE:
  frame_len: 8                        #采样帧数
  sample_freq: 7                      #采样频率
  enable: True                        #开启车辆逆行判断功能
  filter_horizontal_flag: False       #是否过滤水平方向车辆
  keep_right_flag: True               #按车辆靠右行驶规则，若车辆靠左行驶，则设为False
  deviation: 23                       #过滤水平方向车辆的角度阈值，如果大于该角度则过滤
  move_scale: 0.01                    #过滤静止车辆阈值，若车辆移动像素大于图片对角线*move_scale，则认为车辆移动，反之
                                      #车辆静止
  fence_line: []                      #车道中间线坐标，格式[x1,y1,x2,y2] 且y2>y1。若为空，由程序根据车流方向自动判断
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
2. 修改配置文件中`VEHICLE_RETROGRADE`项的`enable: True`，以启用该功能。



3. 车辆逆行识别功能需要视频输入时，启动命令如下：

```bash
#预测单个视频文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_RETROGRADE.enable=true \
                                   --video_file=test_video.mp4 \
                                   --device=gpu

#预测包含一个或多个视频的文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_RETROGRADE.enable=true \
                                   --video_dir=test_video \
                                   --device=gpu
```

5. 若修改模型路径，有以下两种方式：

    - 方法一：`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`下可以配置不同模型路径，车道线识别模型修改`LANE_SEG`字段下配置
    - 方法二：直接在命令行中增加`-o`，以覆盖配置文件中的默认模型路径：

```bash
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   --video_file=test_video.mp4 \
                                   --device=gpu \
                                   -o LANE_SEG.model_dir=output_inference/
                                   VEHICLE_RETROGRADE.enable=true

```
测试效果如下：

<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde.gif"/>
</div>

**注意:**
 - 车道线中间线自动判断条件：在采样的视频段内同时有两个相反方向的车辆，且判断一次后固定，不再更新；
 - 因摄像头角度以及2d视角问题，车道线中间线判断存在不准确情况;
 - 可在配置文件手动输入中间线坐标.参考[车辆违章配置文件](../../config/examples/infer_cfg_vehicle_violation.yml)


## 方案说明
1.车辆在采样视频段内，根据车道中间线的位置与车辆轨迹，判断车辆是否逆行，判断流程图：
<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde.png"/>
</div>

2.车道线识别模型使用了[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 的超轻量分割方案。训练样本[标签](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)分为4类：
  0 背景
  1 双黄线
  2 实线
  3 虚线
车辆逆行分析过滤虚线类；

3.车道线通过对分割结果聚类得到，且默认过滤水平方向车道线，若不过滤可在[车道线配置文件](../../config/lane_seg_config.yml)修改`filter_flag`参数;

4.车辆逆行判断默认过滤水平方向车辆，若不过滤可在[配置文件](../../config/infer_cfg_ppvehicle.yml)修改`filter_horizontal_flag`参数;

5.车辆逆行默认按靠右行驶规则判断，若修改，可在[配置文件](../../config/infer_cfg_ppvehicle.yml)修改`keep_right_flag`参数;

**性能优化措施**：
1.因摄像头视角原因，可以根据实际情况决定是否过滤水平方向车道线与水平方向车辆;

2.车道中间线可手动输入；
