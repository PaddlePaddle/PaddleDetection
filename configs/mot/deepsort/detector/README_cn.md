简体中文 | [English](README.md)

# DeepSORT的检测器

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 由检测器和ReID模型串联组合而成，此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练epoch数、NMS阈值设置等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。

## 模型库

### 在MOT17-half val数据集上的检测结果
| 骨架网络         | 网络类型          |   输入尺度   | 学习率策略    |推理时间(fps)   |  Box AP |   下载    | 配置文件 |
| :-------------- | :-------------  | :--------:  | :---------: | :-----------: | :-----: | :------: | :-----: |
| DarkNet-53      | YOLOv3          |   608X608   |   40e      |      ----     |  42.7   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/yolov3_darknet53_40e_608x608_mot17half.pdparams)  | [配置文件](./yolov3_darknet53_40e_608x608_mot17half.yml) |
| ResNet50-vd     | PPYOLOv2        |   640x640   |   365e      |      ----     |  46.8   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyolov2_r50vd_dcn_365e_640x640_mot17half.pdparams)  | [配置文件](./ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml) |
| CSPResNet       | PPYOLOe         |   640x640   |   36e       |      ----     |  52.9   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyoloe_crn_l_36e_640x640_mot17half.pdparams)     | [配置文件](./ppyoloe_crn_l_36e_640x640_mot17half.yml)    |

**注意:**
  - 以上模型均可采用**MOT17-half train**数据集训练，数据集可以从[此链接](https://dataset.bj.bcebos.com/mot/MOT17.zip)下载。
  - **MOT17-half train**是MOT17的train序列(共7个)每个视频的前一半帧的图片和标注组成的数据集，而为了验证精度可以都用**MOT17-half val**数据集去评估，它是每个视频的后一半帧组成的，数据集可以从[此链接](https://paddledet.bj.bcebos.com/data/mot/mot17half/annotations.zip)下载，并解压放在`dataset/mot/MOT17/images/`文件夹下。
  - YOLOv3和`configs/pedestrian/pedestrian_yolov3_darknet.yml`是相同的pedestrian数据集训练的，此数据集暂未开放。
  - 行人跟踪请使用行人检测器结合行人ReID模型。车辆跟踪请使用车辆检测器结合车辆ReID模型。
  - 用于DeepSORT跟踪时需要高质量的检出框，因此这些模型的NMS阈值等后处理设置会与纯检测任务的设置不同。


## 快速开始

通过如下命令一键式启动训练和评估
```bash
job_name=ppyolov2_r50vd_dcn_365e_640x640_mot17half
config=configs/mot/deepsort/detector/${job_name}.yml
log_dir=log_dir/${job_name}
# 1. training
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp --fleet
# 2. evaluation
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/${job_name}.pdparams
```
