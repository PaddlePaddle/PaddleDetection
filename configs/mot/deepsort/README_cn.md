简体中文 | [English](README.md)

# DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [快速开始](#快速开始)

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442) 与SORT基本类似，但增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征。我们使用JDE作为检测模型来生成检测框，并选择`PCBPyramid`作为ReID模型。我们还支持加载保存的检测结果文件来进行预测跟踪。

## 模型库与基线

### DeepSORT on MOT-16 training set

|  骨干网络  | 输入尺寸 | MOTA |  IDF1  |  IDS | FP  |   FN  |   FPS  | 检测模型 | ReID模型 | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:---: | :---: | :---: |
| DarkNet53 | 1088x608 |  72.2  |  60.3  | 998  |  8055  | 21631 |  3.28 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**Notes:**
  DeepSORT此处不需要训练，只用于评估。在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，这里我们使用JDE，然后像这样准备好结果文件:
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

## 快速开始

### 1. 验证检测模型得到检测结果文件

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o metric=MOT weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams --output ./det_results_dir

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o metric=MOT weights=output/jde_darknet53_30e_1088x608/model_final --output ./det_results_dir
```

### 2. 跟踪预测

```bash
# 加载检测结果文件得到跟踪结果
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml --det_results_dir ./det_results_dir/mot_results
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
