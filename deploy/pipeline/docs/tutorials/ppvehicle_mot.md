[English](ppvehicle_mot_en.md) | 简体中文

# PP-Vehicle车辆跟踪模块

【应用介绍】
车辆检测与跟踪在交通监控、自动驾驶等方向都具有广泛应用，PP-Vehicle中集成了检测跟踪模块，是车牌检测、车辆属性识别等任务的基础。我们提供了预训练模型，用户可以直接下载使用。

【模型下载】
| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 车辆检测/跟踪    |  PP-YOLOE-l | mAP: 63.9 <br> MOTA: 50.1 | 检测: 25.1ms <br> 跟踪：31.8ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| 车辆检测/跟踪    |  PP-YOLOE-s | mAP: 61.3 <br> MOTA: 46.8 | 检测: 16.2ms <br> 跟踪：21.0ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) |

1. 检测/跟踪模型精度为PPVehicle数据集训练得到，整合了BDD100K-MOT和UA-DETRAC，是将BDD100K-MOT中的`car, truck, bus, van`和UA-DETRAC中的`car, bus, van`都合并为1类`vehicle(1)`后的数据集，检测精度mAP是PPVehicle的验证集上测得，跟踪精度MOTA是在BDD100K-MOT的验证集上测得(`car, truck, bus, van`合并为1类`vehicle`)。训练具体流程请参照[ppvehicle](../../../../configs/ppvehicle)。
2. 预测速度为T4 机器上使用TensorRT FP16时的速度, 速度包含数据预处理、模型预测、后处理全流程。

## 使用方法

【配置项说明】

配置文件中与属性相关的参数如下：
```
DET:
  model_dir: output_inference/mot_ppyoloe_l_36e_ppvehicle/ # 车辆检测模型调用路径
  batch_size: 1   # 模型预测时的batch_size大小

MOT:
  model_dir: output_inference/mot_ppyoloe_l_36e_ppvehicle/ # 车辆跟踪模型调用路径
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1   # 模型预测时的batch_size大小, 跟踪任务只能设置为1
  skip_frame_num: -1  # 跳帧预测的帧数，-1表示不进行跳帧，建议跳帧帧数最大不超过3
  enable: False   # 是否开启该功能，使用跟踪前必须确保设置为True
```

【使用命令】
1. 从上表链接中下载模型并解压到```./output_inference```路径下，并修改配置文件中模型路径。默认为自动下载模型，无需做改动。
2. 图片输入时，是纯检测任务，启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. 视频输入时，是跟踪任务，注意首先设置infer_cfg_ppvehicle.yml中的MOT配置的`enable=True`，如果希望跳帧加速检测跟踪流程，可以设置`skip_frame_num: 2`，建议跳帧帧数最大不超过3：
```
MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  skip_frame_num: 2
  enable: True
```
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. 若修改模型路径，有以下两种方式：
    - 方法一：```./deploy/pipeline/config/infer_cfg_ppvehicle.yml```下可以配置不同模型路径，检测和跟踪模型分别对应`DET`和`MOT`字段，修改对应字段下的路径为实际期望的路径即可。
    - 方法二：命令行中--config配置项后面增加`-o MOT.model_dir=[YOUR_DETMODEL_PATH]`修改模型路径。
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --region_type=horizontal \
                                                   --do_entrance_counting \
                                                   --draw_center_traj \
                                                   -o MOT.model_dir=ppyoloe/

```
**注意:**
 - `--do_entrance_counting`表示是否统计出入口流量，不设置即默认为False。
 - `--draw_center_traj`表示是否绘制跟踪轨迹，不设置即默认为False。注意绘制跟踪轨迹的测试视频最好是静止摄像头拍摄的。
 - `--region_type`表示流量计数的区域，当设置`--do_entrance_counting`时可选择`horizontal`或者`vertical`，默认是`horizontal`，表示以视频图片的中心水平线为出入口，同一物体框的中心点在相邻两秒内分别在区域中心水平线的两侧，即完成计数加一。


5. 区域闯入判断和计数

注意首先设置infer_cfg_ppvehicle.yml中的MOT配置的enable=True，然后启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --draw_center_traj \
                                                   --do_break_in_counting \
                                                   --region_type=custom \
                                                   --region_polygon 200 200 400 200 300 400 100 400
```
**注意:**
 - 区域闯入的测试视频必须是静止摄像头拍摄的，镜头不能抖动或移动。
 - `--do_break_in_counting`表示是否进行区域出入后计数，不设置即默认为False。
 - `--region_type`表示流量计数的区域，当设置`--do_break_in_counting`时仅可选择`custom`，默认是`custom`，表示以用户自定义区域为出入口，同一物体框的下边界中点坐标在相邻两秒内从区域外到区域内，即完成计数加一。
 - `--region_polygon`表示用户自定义区域的多边形的点坐标序列，每两个为一对点坐标(x,y)，**按顺时针顺序**连成一个**封闭区域**，至少需要3对点也即6个整数，默认值是`[]`，需要用户自行设置点坐标，如是四边形区域，坐标顺序是`左上、右上、右下、左下`。用户可以运行[此段代码](../../tools/get_video_info.py)获取所测视频的分辨率帧数，以及可以自定义画出自己想要的多边形区域的可视化并自己调整。
 自定义多边形区域的可视化代码运行如下：
 ```python
 python get_video_info.py --video_file=demo.mp4 --region_polygon 200 200 400 200 300 400 100 400
 ```
 快速画出想要的区域的小技巧：先任意取点得到图片，用画图工具打开，鼠标放到想要的区域点上会显示出坐标，记录下来并取整，作为这段可视化代码的region_polygon参数，并再次运行可视化，微调点坐标参数直至满意。


【效果展示】

<div width="600" align="center">
  <img src="../images/mot_vehicle.gif"/>
</div>

## 方案说明

【实现方案及特色】
1. 使用目标检测/多目标跟踪技术来获取图片/视频输入中的车辆检测框，检测模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../../configs/ppyoloe)和[ppvehicle](../../../../configs/ppvehicle)。
2. 多目标跟踪模型方案采用[OC-SORT](https://arxiv.org/pdf/2203.14360.pdf)，采用PP-YOLOE替换原文的YOLOX作为检测器，采用OCSORTTracker作为跟踪器，详细文档参考[OC-SORT](../../../../configs/mot/ocsort)。

## 参考文献
```
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
