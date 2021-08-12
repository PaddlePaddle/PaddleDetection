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

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----:| :-----: | :-----: |
| ResNet-101 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  68.3  |  56.5  | 1722 |  17337 | 15890 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/jde_yolov3_darknet53_30e_1088x608.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

### DeepSORT在MOT-16 Test Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----: | :-----: |:-----: |
| ResNet-101 | 1088x608 |  64.1  |  53.0  | 1024  |  12457  | 51919 |  - | [检测结果](https://dataset.bj.bcebos.com/mot/det_results_dir.zip) | [ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  61.2  |  48.5  | 1799  |  25796  | 43232 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/jde_yolov3_darknet53_30e_1088x608.pdparams)  |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**注意:**

DeepSORT不需要训练MOT数据集，只用于评估，现在支持两种评估的方式。

- 第1种方式是加载检测结果文件和ReID模型，在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，然后像这样准备好结果文件:
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
对于MOT16数据集，可以下载PaddleDetection提供的一个经过匹配之后的检测框结果det_results_dir.zip并解压：
```
wget https://dataset.bj.bcebos.com/mot/det_results_dir.zip
```
如果使用更强的检测模型，可以取得更好的结果。其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id],[bb_left],[bb_top],[width],[height],[conf]
```
- `frame_id`是图片帧的序号
- `bb_left`是目标框的左边界的x坐标
- `bb_top`是目标框的上边界的y坐标
- `width,height`是真实的像素宽高
- `conf`是目标得分设置为`1`(已经按检测的得分阈值筛选出的检测结果)

- 第2种方式是同时加载检测模型和ReID模型，此处选用JDE版本的YOLOv3，具体配置见`configs/mot/deepsort/_base_/deepsort_yolov3_darknet53_pcb_pyramid_r101.yml`

## 快速开始

### 1. 评估

```bash
# 加载检测结果文件和ReID模型，得到跟踪结果
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml --det_results_dir {your detection results}

# 加载检测模型和ReID模型，得到跟踪结果
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_yolov3_pcb_pyramid_r101.yml
```

### 2. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 加载检测模型和ReID模型，得到跟踪结果
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/deepsort/deepsort_yolov3_pcb_pyramid_r101.yml --video_file={your video name}.mp4  --save_videos
```

**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 3. 导出预测模型

```bash
1.先导出检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/jde_yolov3_darknet53_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_yolov3_darknet53_30e_1088x608.pdparams

2.再导出ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/deepsort_yolov3_pcb_pyramid_r101.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams

或
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams
```

### 4. 用导出的模型基于Python去预测

```bash
python deploy/python/mot_sde_infer.py --model_dir=output_inference/jde_yolov3_darknet53_30e_1088x608/ --reid_model_dir=output_inference/deepsort_yolov3_pcb_pyramid_r101/ --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。

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
