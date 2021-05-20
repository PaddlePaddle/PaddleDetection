简体中文 | [English](README.md)

# DeepSORT

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442) (Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402) (Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息。我们使用`JDE`作为检测模型来生成检测框，并选择`PCBPyramid`作为ReID模型。我们还支持加载保存的检测结果文件来进行预测跟踪。

## 模型库

### DeepSORT在MOT-16 train集上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测模型 | ReID模型 | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:-----: | :-----: | :-----: |
| DarkNet53 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  5.07 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**注意:**
  DeepSORT此处不需要训练MOT数据集，只用于评估。在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，此处使用JDE，然后像这样准备好结果文件:
```
det_results_dir
   |——————MOT16-02.txt
   |——————MOT16-04.txt
   |——————MOT16-05.txt
   |——————MOT16-09.txt
   |——————MOT16-10.txt
   |——————MOT16-11.txt
   |——————MOT16-13.txt
```
其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id],[identity],[bb_left],[bb_top],[width],[height],[conf],[x],[y],[z]
```
**注意**:
- `frame_id`是图片帧的序号
- `identity`是目标id采用默认值为`-1`
- `bb_left`是目标框的左边界的x坐标
- `bb_top`是目标框的上边界的y坐标
- `width,height`是真实的像素宽高
- `conf`是目标得分设置为`1`(已经按检测的得分阈值筛选出的检测结果)
- `x,y,z`是3D中用到的，在2D中默认为`-1`

## 快速开始

### 1. 验证检测模型得到检测结果文件

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o weights=output/jde_darknet53_30e_1088x608/model_final.pdparams
```

### 2. 跟踪预测

```bash
# 加载检测结果文件得到跟踪结果
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml --det_results_dir {your detection results}
```

## 引用
```
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
```
