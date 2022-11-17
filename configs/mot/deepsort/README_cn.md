简体中文 | [English](README.md)

# DeepSORT (Deep Cosine Metric Learning for Person Re-identification)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [适配其他检测器](#适配其他检测器)
- [引用](#引用)

## 简介
[DeepSORT](https://arxiv.org/abs/1812.00442)(Deep Cosine Metric Learning SORT) 扩展了原有的[SORT](https://arxiv.org/abs/1703.07402)(Simple Online and Realtime Tracking)算法，增加了一个CNN模型用于在检测器限定的人体部分图像中提取特征，在深度外观描述的基础上整合外观信息，将检出的目标分配和更新到已有的对应轨迹上即进行一个ReID重识别任务。DeepSORT所需的检测框可以由任意一个检测器来生成，然后读入保存的检测结果和视频图片即可进行跟踪预测。ReID模型此处选择[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`PCB+Pyramid ResNet101`和`PPLCNet`模型。

## 模型库

### DeepSORT在MOT-16 Training Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----:| :-----: | :-----: |
| ResNet-101 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  - | [检测结果](https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./reid/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  68.3  |  56.5  | 1722 |  17337 | 15890 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort_jde_yolov3_pcb_pyramid.yml) |
| PPLCNet    | 1088x608 |  72.2  |  59.5  | 1087  |  8034  | 21481 |  - | [检测结果](https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./reid/deepsort_pplcnet.yml) |
| PPLCNet    | 1088x608 |  68.1  |  53.6  | 1979 |  17446 | 15766 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort_jde_yolov3_pplcnet.yml) |

### DeepSORT在MOT-16 Test Set上结果

|  骨干网络  |  输入尺寸  |  MOTA  |  IDF1  |  IDS |   FP   |   FN  |  FPS | 检测结果或模型 | ReID模型 |配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: | :-----: | :-----: |:-----: |
| ResNet-101 | 1088x608 |  64.1  |  53.0  | 1024  |  12457  | 51919 |  - | [检测结果](https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip) | [ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./reid/deepsort_pcb_pyramid_r101.yml) |
| ResNet-101 | 1088x608 |  61.2  |  48.5  | 1799  |  25796  | 43232 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams)  |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)|[配置文件](./deepsort_jde_yolov3_pcb_pyramid.yml) |
| PPLCNet    | 1088x608 |  64.0  |  51.3  | 1208  |  12697  | 51784 |  - | [检测结果](https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./reid/deepsort_pplcnet.yml) |
| PPLCNet    | 1088x608 |  61.1  |  48.8  | 2010 |  25401 | 43432 |  - | [检测模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams) |[ReID模型](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams)|[配置文件](./deepsort_jde_yolov3_pplcnet.yml) |


### DeepSORT在MOT-17 half Val Set上结果

|  检测训练数据集      |  检测器    |  ReID       |  检测mAP  |  MOTA  |  IDF1  |  FPS | 配置文件 |
| :--------         | :-----     | :----:      |:------:  | :----: |:-----: |:----:|:----:   |
| MIX               | JDE YOLOv3 | PCB Pyramid |  -       |  66.9  |  62.7  |   -    |[配置文件](./deepsort_jde_yolov3_pcb_pyramid.yml) |
| MIX               | JDE YOLOv3 | PPLCNet     |  -       |  66.3  |  62.1  |   -    |[配置文件](./deepsort_jde_yolov3_pplcnet.yml) |
| MOT-17 half train | YOLOv3     | PPLCNet     |  42.7    |  50.2  |  52.4  |   -    |[配置文件](./deepsort_yolov3_pplcnet.yml) |
| MOT-17 half train | PPYOLOv2   | PPLCNet     |  46.8    |  51.8  |  55.8  |   -    |[配置文件](./deepsort_ppyolov2_pplcnet.yml) |
| MOT-17 half train | PPYOLOe    | PPLCNet     |  52.7    |  56.7  |  60.5  |   -    |[配置文件](./deepsort_ppyoloe_pplcnet.yml) |
| MOT-17 half train | PPYOLOe    | ResNet-50   |  52.7    |  56.7  |  64.6  |   -    |[配置文件](./deepsort_ppyoloe_resnet.yml) |

**注意:**
模型权重下载链接在配置文件中的```det_weights```和```reid_weights```，运行验证的命令即可自动下载。
DeepSORT是分离检测器和ReID模型的，其中检测器单独训练MOT数据集，而组装成DeepSORT后只用于评估，现在支持两种评估的方式。
- **方式1**：加载检测结果文件和ReID模型，在使用DeepSORT模型评估之前，应该首先通过一个检测模型得到检测结果，然后像这样准备好结果文件:
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
wget https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip
```
如果使用更强的检测模型，可以取得更好的结果。其中每个txt是每个视频中所有图片的检测结果，每行都描述一个边界框，格式如下：
```
[frame_id],[x0],[y0],[w],[h],[score],[class_id]
```
- `frame_id`是图片帧的序号
- `x0,y0`是目标框的左上角x和y坐标
- `w,h`是目标框的像素宽高
- `score`是目标框的得分
- `class_id`是目标框的类别，如果只有1类则是`0`

- **方式2**：同时加载检测模型和ReID模型，此处选用JDE版本的YOLOv3，具体配置见`configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml`。加载其他通用检测模型可参照`configs/mot/deepsort/deepsort_yoloe_pplcnet.yml`进行修改。

## 快速开始

### 1. 评估

#### 1.1 评估检测效果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/deepsort/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
```

**注意:**
 - 评估检测使用的是```tools/eval.py```, 评估跟踪使用的是```tools/eval_mot.py```。

#### 1.2 评估跟踪效果
**方式1**：加载检测结果文件和ReID模型，得到跟踪结果
```bash
# 下载PaddleDetection提供的MOT16数据集检测结果文件并解压，如需自己使用其他检测器生成请参照这个文件里的格式
wget https://bj.bcebos.com/v1/paddledet/data/mot/det_results_dir.zip

CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/reid/deepsort_pcb_pyramid_r101.yml --det_results_dir det_results_dir
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/reid/deepsort_pplcnet.yml --det_results_dir det_results_dir
```

**方式2**：加载行人检测模型和ReID模型，得到跟踪结果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_jde_yolov3_pplcnet.yml
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_ppyolov2_pplcnet.yml --scaled=True
# 或者
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_ppyoloe_resnet.yml --scaled=True
```
**注意:**
 - JDE YOLOv3行人检测模型是和JDE和FairMOT使用同样的MOT数据集训练的，因此MOTA较高。而其他通用检测模型如PPYOLOv2只使用了MOT17 half数据集训练。
 - JDE YOLOv3模型与通用检测模型如YOLOv3和PPYOLOv2最大的区别是使用了JDEBBoxPostProcess后处理，结果输出坐标没有缩放回原图，而通用检测模型输出坐标是缩放回原图的。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE YOLOv3则为False，如果使用通用检测模型则为True, 默认值是False。
 - 跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置。

### 2. 预测

使用单个GPU通过如下命令预测一个视频，并保存为视频

```bash
# 下载demo视频
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/mot17_demo.mp4

# 加载JDE YOLOv3行人检测模型和PCB Pyramid ReID模型，并保存为视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml --video_file=mot17_demo.mp4  --save_videos

# 或者加载PPYOLOE行人检测模型和PPLCNet ReID模型，并保存为视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/deepsort/deepsort_ppyoloe_pplcnet.yml --video_file=mot17_demo.mp4 --scaled=True --save_videos
```

**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


### 3. 导出预测模型

Step 1：导出检测模型
```bash
# 导出JDE YOLOv3行人检测模型
CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c configs/mot/deepsort/detector/jde_yolov3_darknet53_30e_1088x608_mix.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams

# 或导出PPYOLOE行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
```

Step 2：导出ReID模型
```bash
# 导出PCB Pyramid ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pcb_pyramid_r101.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams

# 或者导出PPLCNet ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pplcnet.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams

# 或者导出ResNet ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_resnet.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_resnet.pdparams
```

### 4. 用导出的模型基于Python去预测

```bash
# 用导出的PPYOLOE行人检测模型和PPLCNet ReID模型
python3.7 deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/ppyoloe_crn_l_36e_640x640_mot17half/ --reid_model_dir=output_inference/deepsort_pplcnet/ --tracker_config=deploy/pptracking/python/tracker_config.yml  --video_file=mot17_demo.mp4 --device=GPU --save_mot_txts --threshold=0.5
```
**注意:**
 - 运行前需要先改动`deploy/pptracking/python/tracker_config.yml`里的tracker为`DeepSORTTracker`。
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示对每个视频保存一个txt，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。


## 适配其他检测器

### 1、配置文件目录说明
- `detector/xxx.yml`是纯粹的检测模型配置文件，如`detector/ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml`，支持检测的所有流程(train/eval/infer/export/deploy)。DeepSORT跟踪的eval/infer与这个纯检测的yml文件无关，但是export的时候需要这个纯检测的yml单独导出检测模型，DeepSORT跟踪导出模型是分开detector和reid分别导出的，用户可自行定义和组装detector+reid成为一个完整的DeepSORT跟踪系统。
- `detector/`下的检测器配置文件中，用户需要将自己的数据集转为COCO格式。由于ID的真实标签不需要参与进去，用户可以在此自行配置任何检测模型，只需保证输出结果包含结果框的种类、坐标和分数即可。
- `reid/deepsort_yyy.yml`文件夹里的是ReID模型和tracker的配置文件，如`reid/deepsort_pplcnet.yml`，此处ReID模型是由[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供的`deepsort_pcb_pyramid_r101.yml`和`deepsort_pplcnet.yml`，是在Market1501(751类人)行人ReID数据集上训练得到的，训练细节待PaddleClas公布。
- `deepsort_xxx_yyy.yml`是一个完整的DeepSORT跟踪的配置，如`deepsort_ppyolov2_pplcnet.yml`，其中检测部分`xxx`是`detector/`里的，reid和tracker部分`yyy`是`reid/`里的。
- DeepSORT跟踪的eval/infer有两种方式，方式1是只使用`reid/deepsort_yyy.yml`加载检测结果文件和`yyy`ReID模型，方式2是使用`deepsort_xxx_yyy.yml`加载`xxx`检测模型和`yyy`ReID模型，但是DeepSORT跟踪的deploy必须使用`deepsort_xxx_yyy.yml`。
- 检测器的eval/infer/deploy只使用到`detector/xxx.yml`，ReID一般不单独使用，如需单独使用必须提前加载检测结果文件然后只使用`reid/deepsort_yyy.yml`。


### 2、适配的具体步骤
1.先将数据集制作成COCO格式按通用检测模型配置来训练，参照`detector/`文件夹里的模型配置文件，制作生成`detector/xxx.yml`, 已经支持有Faster R-CNN、YOLOv3、PPYOLOv2、JDE YOLOv3和PicoDet等模型。

2.制作`deepsort_xxx_yyy.yml`, 其中`DeepSORT.detector`的配置就是`detector/xxx.yml`里的, `EvalMOTDataset`和`det_weights`可以自行设置。`yyy`是`reid/deepsort_yyy.yml`如`reid/deepsort_pplcnet.yml`。

### 3、使用的具体步骤
#### 1.加载检测模型和ReID模型去评估:
```
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_xxx_yyy.yml --scaled=True
```
#### 2.加载检测模型和ReID模型去推理:
```
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/deepsort/deepsort_xxx_yyy.yml --video_file=mot17_demo.mp4 --scaled=True --save_videos
```
#### 3.导出检测模型和ReID模型:
```bash
# 导出检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/detector/xxx.yml
# 导出ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_yyy.yml
```
#### 4.使用导出的检测模型和ReID模型去部署:
```
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/xxx./ --reid_model_dir=output_inference/deepsort_yyy/ --video_file=mot17_demo.mp4 --device=GPU --scaled=True --save_mot_txts
```
**注意:**
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


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
