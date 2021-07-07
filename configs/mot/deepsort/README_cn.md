简体中文 | [English](README.md)

# DeepSORT (Deep Cosine Metric Learning for Person Re-identification)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个ReID重识别任务。DeepSORT所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID模型此处选择[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`PCB+Pyramid ResNet101`模型。

## 模型库

### DeepSORT在MOT-16 Training Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 下载链接 | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----: | :-----: |
| ResNet101 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  - | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

### DeepSORT在MOT-16 Test Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 下载链接 | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----: | :-----: |
| ResNet101 | 1088x608 |  64.1  |  53.0  | 1024  |  12457  | 51919 |  - | [下载链接](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**注意:**
  DeepSORT不需要训练MOT数据集，只用于评估。在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，然后像这样准备好结果文件:
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
对于MOT16数据集，可以下载PaddleDetection提供的det_results_dir.zip并解压：
```
wget https://dataset.bj.bcebos.com/mot/det_results_dir.zip
```
其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id],[identity],[bb_left],[bb_top],[width],[height],[conf]
```
**注意**:
- `frame_id`是图片帧的序号
- `identity`是目标id采用默认值为`-1`
- `bb_left`是目标框的左边界的x坐标
- `bb_top`是目标框的上边界的y坐标
- `width,height`是真实的像素宽高
- `conf`是目标得分设置为`1`(已经按检测的得分阈值筛选出的检测结果)

## 快速开始

### 1. 评估

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
