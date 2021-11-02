简体中文 | [English](README.md)

# DeepSORT的检测器

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 由检测器和ReID模型串联组合而成，此处提供了几个常用检测器的配置作为参考。由于训练数据集、输入尺度、训练epoch数等的不同均会导致模型精度和性能的差异，请自行根据需求进行适配。

## 模型库

### 在MOT17-half val数据集上的检测结果
| 骨架网络         | 网络类型          |   输入尺度   | 学习率策略    |推理时间(fps)   |  Box AP |   下载    | 配置文件 |
| :-------------- | :-------------  | :--------:  | :---------: | :-----------: | :-----: | :------: | :-----: |
| ResNet50-vd     | PPYOLOv2        |   640x640   |   365e      |      ----     |  45.0   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/ppyolov2_r50vd_dcn_365e_640x640_mot17half.pdparams)  | [配置文件](./ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml) |
| ResNet50-FPN    | Faster R-CNN    |   1333x800  |   1x        |      ----     |  42.9   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/faster_rcnn_r50_fpn_2x_1333x800_mot17half.pdparams)  | [配置文件](./faster_rcnn_r50_fpn_2x_1333x800_mot17half.yml) |
| ESNet           | PicoDet         |    896x896  |   300e      |      ----     |  40.4   | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort/detector/picodet_l_esnet_300e_640x640_mot17half.pdparams)     | [配置文件](./picodet_l_esnet_300e_640x640_mot17half.yml)    |

**注意:**
  以上模型采用MOT17-half train数据集训练，是MOT17的train序列(共7个)每个视频的前一半帧的图片和标注组成的数据集，而为了验证精度用MOT17-half val数据集去评估，它是每个视频的后一半帧组成的。
  行人跟踪请使用行人检测器结合行人ReID模型。车辆跟踪请使用车辆检测器结合车辆ReID模型。
