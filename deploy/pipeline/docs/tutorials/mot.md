[English](mot_en.md) | 简体中文

# PP-Human检测跟踪模块

行人检测与跟踪在智慧社区，工业巡检，交通监控等方向都具有广泛应用，PP-Human中集成了检测跟踪模块，是关键点检测、属性行为识别等任务的基础。我们提供了预训练模型，用户可以直接下载使用。

| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 行人检测/跟踪    |  PP-YOLOE-l | mAP: 56.6 <br> MOTA: 79.5 | 检测: 28.0ms <br> 跟踪：33.1ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 行人检测/跟踪    |  PP-YOLOE-s | mAP: 53.2 <br> MOTA: 69.1 | 检测: 22.1ms <br> 跟踪：27.2ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |

1. 检测/跟踪模型精度为[COCO-Person](http://cocodataset.org/), [CrowdHuman](http://www.crowdhuman.org/), [HIEVE](http://humaninevents.org/) 和部分业务数据融合训练测试得到，验证集为业务数据
2. 预测速度为T4 机器上使用TensorRT FP16时的速度, 速度包含数据预处理、模型预测、后处理全流程

## 使用方法

1. 从上表链接中下载模型并解压到```./output_inference```路径下
2. 图片输入时，是纯检测任务，启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. 视频输入时，是跟踪任务，注意首先设置infer_cfg_pphuman.yml中的MOT配置的enable=True，然后启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. 若修改模型路径，有以下两种方式：

    - ```./deploy/pipeline/config/infer_cfg_pphuman.yml```下可以配置不同模型路径，检测和跟踪模型分别对应`DET`和`MOT`字段，修改对应字段下的路径为实际期望的路径即可。
    - 命令行中增加`--model_dir`修改模型路径：
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --region_type=horizontal \
                                                   --do_entrance_counting \
                                                   --draw_center_traj \
                                                   --model_dir det=ppyoloe/

```
**注意:**
 - `--do_entrance_counting`表示是否统计出入口流量，不设置即默认为False。
 - `--draw_center_traj`表示是否绘制跟踪轨迹，不设置即默认为False。注意绘制跟踪轨迹的测试视频最好是静止摄像头拍摄的。
 - `--region_type`表示流量计数的区域，当设置`--do_entrance_counting`时可选择`horizontal`或者`vertical`，默认是`horizontal`，表示以视频图片的中心水平线为出入口，同一物体框的中心点在相邻两秒内分别在区域中心水平线的两侧，即完成计数加一。

测试效果如下：

<div width="1000" align="center">
  <img src="../images/mot.gif"/>
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
                                                   --region_polygon 900 500 1580 500 1790 900 890 900
```
**注意:**
 - `--do_break_in_counting`表示是否进行区域出入后计数，不设置即默认为False。
 - `--region_type`表示流量计数的区域，当设置`--do_break_in_counting`时仅可选择`custom`，默认是`custom`，表示以用户自定义区域为出入口，同一物体框的下边界中点坐标在相邻两秒内从区域外到区域内，即完成计数加一。
 - `--region_polygon`表示用户自定义区域的多边形的点坐标序列，每两个为一对点坐标(x,y坐标)，至少需要3对点也即6个整数。用户可以运行一下代码获取所测视频的分辨率帧数，以及可以自定义画出自己想要的多边形区域的可视化并自己调整。
  <details>

  ```python
  import os
  import sys
  import cv2
  import numpy as np

  video_file = 'demo.mp4'
  region_polygon = [
      [900, 500],
      [1580, 500],
      [1790, 900],
      [890, 900],
  ] # modify by yourself, at least 3 pairs of points
  region_polygon = np.array(region_polygon)

  if not os.path.exists(video_file):
      print("video path '{}' not exists".format(video_file))
      sys.exit(-1)
  capture = cv2.VideoCapture(video_file)
  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print("video width: %d, height: %d" % (width, height))
  np_masks = np.zeros((height, width, 1), np.uint8)
  cv2.fillPoly(np_masks, [region_polygon], 255)

  fps = int(capture.get(cv2.CAP_PROP_FPS))
  frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  print("video fps: %d, frame_count: %d" % (fps, frame_count))
  cnt = 0
  while (1):
      ret, frame = capture.read()
      cnt += 1
      if cnt == 3: break

  alpha = 0.3
  img = np.array(frame).astype('float32')
  mask = np_masks[:, :, 0]
  color_mask = [0, 0, 255]
  idx = np.nonzero(mask)
  color_mask = np.array(color_mask)
  img[idx[0], idx[1], :] *= 1.0 - alpha
  img[idx[0], idx[1], :] += alpha * color_mask
  cv2.imwrite('region_vis.jpg', img)

  points_info = 'Your region_polygon points are:'
  for p in region_polygon:
      points_info += ' {} {}'.format(p[0], p[1])
  ```

  </details>

## 方案说明

1. 目标检测/多目标跟踪获取图片/视频输入中的行人检测框，模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../../configs/ppyoloe/)
2. 多目标跟踪模型方案采用[ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)和[OC-SORT](https://arxiv.org/pdf/2203.14360.pdf)，采用PP-YOLOE替换原文的YOLOX作为检测器，采用BYTETracker和OCSORTTracker作为跟踪器，详细文档参考[ByteTrack](../../../../configs/mot/bytetrack)和[OC-SORT](../../../../configs/mot/ocsort)

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
