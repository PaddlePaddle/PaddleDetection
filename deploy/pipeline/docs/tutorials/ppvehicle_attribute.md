[English](ppvehicle_attribute_en.md) | 简体中文

# PP-Vehicle属性识别模块

车辆属性识别在智慧城市，智慧交通等方向具有广泛应用。在PP-Vehicle中，集成了车辆属性识别模块，可识别车辆颜色及车型属性的识别。

| 任务 | 算法 | 精度 | 预测速度 | 下载链接|
|-----------|------|-----------|----------|---------------|
| 车辆检测/跟踪 | PP-YOLOE | mAP 63.9 | 38.67ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| 车辆属性识别 | PPLCNet | 90.81 | 7.31 ms | [预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip) |


注意：
1. 属性模型预测速度是基于NVIDIA T4, 开启TensorRT FP16得到。模型预测速度包含数据预处理、模型预测、后处理部分。
2. 关于PP-LCNet的介绍可以参考[PP-LCNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNet.md)介绍，相关论文可以查阅[PP-LCNet paper](https://arxiv.org/abs/2109.15099)。
3. 属性模型的训练和精度测试均基于[VeRi数据集](https://www.v7labs.com/open-datasets/veri-dataset)。


- 当前提供的预训练模型支持识别10种车辆颜色及9种车型，同VeRi数据集，具体如下：

```yaml
# 车辆颜色
- "yellow"
- "orange"
- "green"
- "gray"
- "red"
- "blue"
- "white"
- "golden"
- "brown"
- "black"

# 车型
- "sedan"
- "suv"
- "van"
- "hatchback"
- "mpv"
- "pickup"
- "bus"
- "truck"
- "estate"
```

## 使用方法

### 配置项说明

[配置文件](../../config/infer_cfg_ppvehicle.yml)中与属性相关的参数如下：
```
VEHICLE_ATTR:
  model_dir: output_inference/vehicle_attribute_infer/ # 车辆属性模型调用路径
  batch_size: 8     # 模型预测时的batch_size大小
  color_threshold: 0.5  # 颜色属性阈值，需要置信度达到此阈值才会确定具体颜色，否则为'Unknown‘
  type_threshold: 0.5   # 车型属性阈值，需要置信度达到此阈值才会确定具体属性，否则为'Unknown‘
  enable: False         # 是否开启该功能
```

### 使用命令

1. 从模型库下载`车辆检测/跟踪`, `车辆属性识别`两个预测部署模型并解压到`./output_inference`路径下；默认会自动下载模型，如果手动下载，需要修改模型文件夹为模型存放路径。
2. 修改配置文件中`VEHICLE_ATTR`项的`enable: True`，以启用该功能。
3. 图片输入时，启动命令如下(更多命令参数说明，请参考[快速开始-参数说明](./PPVehicle_QUICK_STARTED.md))：

```bash
# 预测单张图片文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu

# 预测包含一张或多张图片的文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --image_dir=images/ \
                                                   --device=gpu
```

4. 视频输入时，启动命令如下：

```bash
#预测单个视频文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu

#预测包含一个或多个视频的文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_dir=test_videos/ \
                                                   --device=gpu
```

5. 若修改模型路径，有以下两种方式：

    - 方法一：`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`下可以配置不同模型路径，属性识别模型修改`VEHICLE_ATTR`字段下配置
    - 方法二：直接在命令行中增加`-o`，以覆盖配置文件中的默认模型路径：

```bash
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   -o VEHICLE_ATTR.model_dir=output_inference/vehicle_attribute_infer
```

测试效果如下：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205599146-56abd72f-6e0a-4a21-bd11-f8bb421f2887.gif"/>
</div>

## 方案说明
车辆属性识别模型使用了[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 的超轻量图像分类方案(PULC，Practical Ultra Lightweight image Classification)。关于该模型的数据准备、训练、测试等详细内容，请见[PULC 车辆属性识别模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/PULC/PULC_vehicle_attribute.md).

车辆属性识别模型选用了轻量级、高精度的PPLCNet。并在该模型的基础上，进一步使用了以下优化方案：

- 使用SSLD预训练模型，在不改变推理速度的前提下，精度可以提升约0.5个百分点
- 融合EDA数据增强策略，精度可以再提升0.52个百分点
- 使用SKL-UGI知识蒸馏, 精度可以继续提升0.23个百分点
