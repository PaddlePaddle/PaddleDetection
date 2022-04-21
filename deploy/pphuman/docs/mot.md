# PP-Human检测跟踪模块

行人检测与跟踪在智慧社区，工业巡检，交通监控等方向都具有广泛应用，PP-Human中集成了检测跟踪模块，是关键点检测、属性行为识别等任务的基础。我们提供了预训练模型，用户可以直接下载使用。

| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 行人检测/跟踪    |  PP-YOLOE | mAP: 56.3 <br> MOTA: 72.0 | 检测: 28ms <br> 跟踪：33.1ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |

1. 检测/跟踪模型精度为MOT17，CrowdHuman，HIEVE和部分业务数据融合训练测试得到
2. 预测速度为T4 机器上使用TensorRT FP16时的速度, 速度包含数据预处理、模型预测、后处理全流程

## 使用方法

1. 从上表链接中下载模型并解压到```./output_inference```路径下
2. 图片输入时，启动命令如下
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. 视频输入时，启动命令如下
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. 若修改模型路径，有以下两种方式：

    - ```./deploy/pphuman/config/infer_cfg.yml```下可以配置不同模型路径，检测和跟踪模型分别对应`DET`和`MOT`字段，修改对应字段下的路径为实际期望的路径即可。
    - 命令行中增加`--model_dir`修改模型路径：
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --model_dir det=ppyoloe/
                                                   --do_entrance_counting \
                                                   --draw_center_traj

```
**注意:**
 - `--do_entrance_counting`表示是否统计出入口流量，不设置即默认为False
 - `--draw_center_traj`表示是否绘制跟踪轨迹，不设置即默认为False。注意绘制跟踪轨迹的测试视频最好是静止摄像头拍摄的。

测试效果如下：

<div width="1000" align="center">
  <img src="./images/mot.gif"/>
</div>

数据来源及版权归属：天覆科技，感谢提供并开源实际场景数据，仅限学术研究使用

## 方案说明

1. 目标检测/多目标跟踪获取图片/视频输入中的行人检测框，模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../configs/ppyoloe/README_cn.md)
2. 多目标跟踪模型方案基于[ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)，采用PP-YOLOE替换原文的YOLOX作为检测器，采用BYTETracker作为跟踪器。

## 参考文献
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
