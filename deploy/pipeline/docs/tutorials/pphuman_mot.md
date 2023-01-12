[English](pphuman_mot_en.md) | 简体中文

# PP-Human检测跟踪模块

行人检测与跟踪在智慧社区，工业巡检，交通监控等方向都具有广泛应用，PP-Human中集成了检测跟踪模块，是关键点检测、属性行为识别等任务的基础。我们提供了预训练模型，用户可以直接下载使用。

| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 行人检测/跟踪    |  PP-YOLOE-l | mAP: 57.8 <br> MOTA: 82.2 | 检测: 25.1ms <br> 跟踪：31.8ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 行人检测/跟踪    |  PP-YOLOE-s | mAP: 53.2 <br> MOTA: 73.9 | 检测: 16.2ms <br> 跟踪：21.0ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |

1. 检测/跟踪模型精度为[COCO-Person](http://cocodataset.org/), [CrowdHuman](http://www.crowdhuman.org/), [HIEVE](http://humaninevents.org/) 和部分业务数据融合训练测试得到，验证集为业务数据
2. 预测速度为T4 机器上使用TensorRT FP16时的速度, 速度包含数据预处理、模型预测、后处理全流程

## 使用方法

1. 从上表链接中下载模型并解压到```./output_inference```路径下，并修改配置文件中模型路径。默认为自动下载模型，无需做改动。
2. 图片输入时，是纯检测任务，启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. 视频输入时，是跟踪任务，注意首先设置infer_cfg_pphuman.yml中的MOT配置的`enable=True`，如果希望跳帧加速检测跟踪流程，可以设置`skip_frame_num: 2`，建议跳帧帧数最大不超过3：
```
MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  skip_frame_num: 2
  enable: True
```
然后启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. 若修改模型路径，有以下两种方式：

    - ```./deploy/pipeline/config/infer_cfg_pphuman.yml```下可以配置不同模型路径，检测和跟踪模型分别对应`DET`和`MOT`字段，修改对应字段下的路径为实际期望的路径即可。
    - 命令行中--config后面紧跟着增加`-o MOT.model_dir`修改模型路径：
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   -o MOT.model_dir=ppyoloe/\
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --region_type=horizontal \
                                                   --do_entrance_counting \
                                                   --draw_center_traj

```
**注意:**
 - `--do_entrance_counting`表示是否统计出入口流量，不设置即默认为False。
 - `--draw_center_traj`表示是否绘制跟踪轨迹，不设置即默认为False。注意绘制跟踪轨迹的测试视频最好是静止摄像头拍摄的。
 - `--region_type`表示流量计数的区域，当设置`--do_entrance_counting`时可选择`horizontal`或者`vertical`，默认是`horizontal`，表示以视频图片的中心水平线为出入口，同一物体框的中心点在相邻两秒内分别在区域中心水平线的两侧，即完成计数加一。

测试效果如下：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205599943-8da89ce8-f6d1-47e5-adc8-6d199b76d167.gif"/>
</div>

数据来源及版权归属：天覆科技，感谢提供并开源实际场景数据，仅限学术研究使用

5. 区域闯入判断和计数

注意首先设置infer_cfg_pphuman.yml中的MOT配置的enable=True，然后启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
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


测试效果如下：

<div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/178769370-03ab1965-cfd1-401b-9902-82620a06e43c.gif" width='600'/>
</div>

## 方案说明

1. 使用目标检测/多目标跟踪技术来获取图片/视频输入中的行人检测框，检测模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../../configs/ppyoloe)。
2. 多目标跟踪模型方案采用[ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)和[OC-SORT](https://arxiv.org/pdf/2203.14360.pdf)，采用PP-YOLOE替换原文的YOLOX作为检测器，采用BYTETracker和OCSORTTracker作为跟踪器，详细文档参考[ByteTrack](../../../../configs/mot/bytetrack)和[OC-SORT](../../../../configs/mot/ocsort)。

## 参考文献
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}

@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
