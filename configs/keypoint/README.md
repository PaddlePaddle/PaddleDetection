简体中文 | [English](README_en.md)

# 关键点检测系列模型

<div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205551833-a891a790-73c6-43cb-84f9-91553e9ef27b.gif" width='800'/>
</div>

## 目录

- [简介](#简介)
- [模型推荐](#模型推荐)
- [模型库](#模型库)
- [快速开始](#快速开始)
  - [环境安装](#1环境安装)
  - [数据准备](#2数据准备)
  - [训练与测试](#3训练与测试)
    - [单卡训练](#单卡训练)
    - [多卡训练](#多卡训练)
    - [模型评估](#模型评估)
    - [模型预测](#模型预测)
    - [模型部署](#模型部署)
      - [Top-Down模型联合部署](#top-down模型联合部署)
      - [Bottom-Up模型独立部署](#bottom-up模型独立部署)
      - [与多目标跟踪联合部署](#与多目标跟踪模型fairmot联合部署)
    - [完整部署教程及Demo](#完整部署教程及Demo)
- [自定义数据训练](#自定义数据训练)
- [BenchMark](#benchmark)

## 简介

PaddleDetection 中的关键点检测部分紧跟最先进的算法，包括 Top-Down 和 Bottom-Up 两种方法，可以满足用户的不同需求。Top-Down 先检测对象，再检测特定关键点。Top-Down 模型的准确率会更高，但速度会随着对象数量的增加而变慢。不同的是，Bottom-Up 首先检测点，然后对这些点进行分组或连接以形成多个人体姿势实例。Bottom-Up 的速度是固定的，不会随着物体数量的增加而变慢，但精度会更低。

同时，PaddleDetection 提供针对移动端设备优化的自研实时关键点检测模型 [PP-TinyPose](./tiny_pose/README.md)。

## 模型推荐

### 移动端模型推荐

| 检测模型                                                     | 关键点模型                            |             输入尺寸             |         COCO数据集精度          |          平均推理耗时 (FP16)           | 参数量 （M）                |          Flops (G)          |                           模型权重                           |                  Paddle-Lite部署模型（FP16)                  |
| :----------------------------------------------------------- | :------------------------------------ | :------------------------------: | :-----------------------------: | :------------------------------------: | --------------------------- | :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [PicoDet-S-Pedestrian](../picodet/legacy_model/application/pedestrian_detection/picodet_s_192_pedestrian.yml) | [PP-TinyPose](./tiny_pose/tinypose_128x96.yml)  | 检测：192x192<br>关键点：128x96  | 检测mAP：29.0<br>关键点AP：58.1 | 检测耗时：2.37ms<br>关键点耗时：3.27ms | 检测：1.18<br/>关键点：1.36 | 检测：0.35<br/>关键点：0.08 | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian.pdparams)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian_fp16.nb)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_fp16.nb) |
| [PicoDet-S-Pedestrian](../picodet/legacy_model/application/pedestrian_detection/picodet_s_320_pedestrian.yml) | [PP-TinyPose](./tiny_pose/tinypose_256x192.yml) | 检测：320x320<br>关键点：256x192 | 检测mAP：38.5<br>关键点AP：68.8 | 检测耗时：6.30ms<br>关键点耗时：8.33ms | 检测：1.18<br/>关键点：1.36 | 检测：0.97<br/>关键点：0.32 | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.pdparams)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) | [检测](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian_fp16.nb)<br>[关键点](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_fp16.nb) |


*详细关于PP-TinyPose的使用请参考[文档](./tiny_pose/README.md)。

### 服务端模型推荐

| 检测模型                                                     | 关键点模型                                 |             输入尺寸             |         COCO数据集精度          |       参数量 （M）       |        Flops (G)         |                           模型权重                           |
| :----------------------------------------------------------- | :----------------------------------------- | :------------------------------: | :-----------------------------: | :----------------------: | :----------------------: | :----------------------------------------------------------: |
| [PP-YOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [HRNet-w32](./hrnet/hrnet_w32_384x288.yml) | 检测：640x640<br>关键点：384x288 | 检测mAP：49.5<br>关键点AP：77.8 | 检测：54.6<br/>关键点：28.6 | 检测：115.8<br/>关键点：17.3 | [检测](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams)<br>[关键点](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) |
| [PP-YOLOv2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [HRNet-w32](./hrnet/hrnet_w32_256x192.yml) | 检测：640x640<br>关键点：256x192 | 检测mAP：49.5<br>关键点AP：76.9 | 检测：54.6<br/>关键点：28.6 | 检测：115.8<br/>关键点：7.68 | [检测](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams)<br>[关键点](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) |


## 模型库

COCO数据集

| 模型              |  方案              |输入尺寸 | AP(coco val) |                           模型下载                           | 配置文件 |  
| :---------------- | -------- | :----------: | :----------------------------------------------------------: | ----------------------------------------------------| ------- |
| PETR_Res50       |One-Stage| 512      |     65.5     | [petr_res50.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/petr_resnet50_16x2_coco.pdparams) | [config](./petr/petr_resnet50_16x2_coco.yml)       |
| HigherHRNet-w32       |Bottom-Up| 512      |     67.1     | [higherhrnet_hrnet_w32_512.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512.yml)       |
| HigherHRNet-w32       | Bottom-Up| 640      |     68.3     | [higherhrnet_hrnet_w32_640.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_640.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_640.yml)       |
| HigherHRNet-w32+SWAHR |Bottom-Up|  512      |     68.9     | [higherhrnet_hrnet_w32_512_swahr.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512_swahr.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512_swahr.yml) |
| HRNet-w32             | Top-Down| 256x192  |     76.9     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](./hrnet/hrnet_w32_256x192.yml)                     |
| HRNet-w32             |Top-Down| 384x288  |     77.8     | [hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) | [config](./hrnet/hrnet_w32_384x288.yml)                     |
| HRNet-w32+DarkPose             |Top-Down| 256x192  |     78.0     | [dark_hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) | [config](./hrnet/dark_hrnet_w32_256x192.yml)                     |
| HRNet-w32+DarkPose             |Top-Down| 384x288  |     78.3     | [dark_hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) | [config](./hrnet/dark_hrnet_w32_384x288.yml)                     |
| WiderNaiveHRNet-18         | Top-Down|256x192  |     67.6(+DARK 68.4)     | [wider_naive_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/wider_naive_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/wider_naive_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   |Top-Down| 256x192  |     66.5     | [lite_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   |Top-Down| 384x288  |     69.7     | [lite_hrnet_18_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_384x288_coco.yml)     |
| LiteHRNet-30                   | Top-Down|256x192  |     69.4     | [lite_hrnet_30_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_256x192_coco.yml)     |
| LiteHRNet-30                   |Top-Down| 384x288  |     72.5     | [lite_hrnet_30_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_384x288_coco.yml)     |

备注： Top-Down模型测试AP结果基于GroundTruth标注框

MPII数据集
| 模型  | 方案| 输入尺寸 | PCKh(Mean) | PCKh(Mean@0.1) |                           模型下载                           | 配置文件                                     |
| :---- | ---|----- | :--------: | :------------: | :----------------------------------------------------------: | -------------------------------------------- |
| HRNet-w32 | Top-Down|256x256  |    90.6    |      38.5      | [hrnet_w32_256x256_mpii.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x256_mpii.pdparams) | [config](./hrnet/hrnet_w32_256x256_mpii.yml) |

场景模型
| 模型 | 方案 | 输入尺寸 | 精度 | 预测速度 |模型权重 | 部署模型 | 说明|
| :---- | ---|----- | :--------: | :--------: | :------------: |:------------: |:-------------------: |
| HRNet-w32 + DarkPose | Top-Down|256x192  |  AP: 87.1 (业务数据集)| 单人2.9ms |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.pdparams) |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) | 针对摔倒场景特别优化，该模型应用于[PP-Human](../../deploy/pipeline/README.md) |


我们同时推出了基于LiteHRNet（Top-Down）针对移动端设备优化的实时关键点检测模型[PP-TinyPose](./tiny_pose/README.md), 欢迎体验。



## 快速开始

### 1、环境安装

​    请参考PaddleDetection [安装文档](../../docs/tutorials/INSTALL_cn.md)正确安装PaddlePaddle和PaddleDetection即可。

### 2、数据准备

​    目前KeyPoint模型支持[COCO](https://cocodataset.org/#keypoints-2017)数据集和[MPII](http://human-pose.mpi-inf.mpg.de/#overview)数据集，数据集的准备方式请参考[关键点数据准备](../../docs/tutorials/data/PrepareKeypointDataSet.md)。

​    关于config配置文件内容说明请参考[关键点配置文件说明](../../docs/tutorials/KeyPointConfigGuide_cn.md)。

- 请注意，Top-Down方案使用检测框测试时，需要通过检测模型生成bbox.json文件。COCO val2017的检测结果可以参考[Detector having human AP of 56.4 on COCO val2017 dataset](https://paddledet.bj.bcebos.com/data/bbox.json)，下载后放在根目录（PaddleDetection）下，然后修改config配置文件中`use_gt_bbox: False`后生效。然后正常执行测试命令即可。

### 3、训练与测试

#### 单卡训练

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

#### 多卡训练

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

#### 模型评估

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml

#当只需要保存评估预测的结果时，可以通过设置save_prediction_only参数实现，评估预测结果默认保存在output/keypoints_results.json文件中
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml --save_prediction_only
```

#### 模型预测

​    注意：top-down模型只支持单人截图预测，如需使用多人图，请使用[联合部署推理]方式。或者使用bottom-up模型。

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=./output/higherhrnet_hrnet_w32_512/model_final.pdparams --infer_dir=../images/ --draw_threshold=0.5 --save_txt=True
```

#### 模型部署

##### Top-Down模型联合部署

```shell
#导出检测模型
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams

#导出关键点模型
python tools/export_model.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml -o weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams

#detector 检测 + keypoint top-down模型联合部署（联合推理只支持top-down方式）
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file=../video/xxx.mp4  --device=gpu
```

##### Bottom-Up模型独立部署

```shell
#导出模型
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=output/higherhrnet_hrnet_w32_512/model_final.pdparams

#部署推理
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --device=gpu --threshold=0.5
```

##### 与多目标跟踪模型FairMOT联合部署

```shell
#导出FairMOT跟踪模型
python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

#用导出的跟踪和关键点模型Python联合预测
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```

**注意:**
 跟踪模型导出教程请参考[文档](../mot/README.md)。

### 完整部署教程及Demo


​ 我们提供了PaddleInference(服务器端)、PaddleLite(移动端)、第三方部署(MNN、OpenVino)支持。无需依赖训练代码，deploy文件夹下相应文件夹提供独立完整部署代码。 详见 [部署文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/README.md)介绍。

## 自定义数据训练

我们以[tinypose_256x192](./tiny_pose/README.md)为例来说明对于自定义数据如何修改：

#### 1、配置文件[tinypose_256x192.yml](../../configs/keypoint/tiny_pose/tinypose_256x192.yml)

基本的修改内容及其含义如下：

```
num_joints: &num_joints 17    #自定义数据的关键点数量
train_height: &train_height 256   #训练图片尺寸-高度h
train_width: &train_width 192   #训练图片尺寸-宽度w
hmsize: &hmsize [48, 64]  #对应训练尺寸的输出尺寸，这里是输入[w,h]的1/4
flip_perm: &flip_perm [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]] #关键点定义中左右对称的关键点，用于flip增强。若没有对称结构在 TrainReader 的 RandomFlipHalfBodyTransform 一栏中 flip_pairs 后面加一行 "flip: False"（注意缩紧对齐）
num_joints_half_body: 8   #半身关键点数量，用于半身增强
prob_half_body: 0.3   #半身增强实现概率，若不需要则修改为0
upper_body_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    #上半身对应关键点id，用于半身增强中获取上半身对应的关键点。
```

上述是自定义数据时所需要的修改部分，完整的配置及含义说明可参考文件：[关键点配置文件说明](../../docs/tutorials/KeyPointConfigGuide_cn.md)。

#### 2、其他代码修改（影响测试、可视化）
- keypoint_utils.py中的sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,.87, .87, .89, .89]) / 10.0，表示每个关键点的确定范围方差，根据实际关键点可信区域设置，区域精确的一般0.25-0.5，例如眼睛。区域范围大的一般0.5-1.0，例如肩膀。若不确定建议0.75。
- visualizer.py中的draw_pose函数中的EDGES，表示可视化时关键点之间的连接线关系。
- pycocotools工具中的sigmas，同第一个keypoint_utils.py中的设置。用于coco指标评估时计算。

#### 3、数据准备注意
- 训练数据请按coco数据格式处理。需要包括关键点[Nx3]、检测框[N]标注。
- 请注意area>0，area=0时数据在训练时会被过滤掉。此外，由于COCO的评估机制，area较小的数据在评估时也会被过滤掉，我们建议在自定义数据时取`area = bbox_w * bbox_h`。

如有遗漏，欢迎反馈


## 关键点稳定策略（仅适用于视频数据）
使用关键点算法处理视频数据时，由于预测针对单帧图像进行，在视频结果上往往会有抖动的现象。在一些依靠精细化坐标的应用场景（例如健身计数、基于关键点的虚拟渲染等）上容易造成误检或体验不佳的问题。针对这个问题，在PaddleDetection关键点视频推理中加入了[OneEuro滤波器](http://www.lifl.fr/~casiez/publications/CHI2012-casiez.pdf)和EMA两种关键点稳定方式。实现将当前关键点坐标结果和历史关键点坐标结果结合计算，使得输出的点坐标更加稳定平滑。该功能同时支持在Python及C++推理中一键开启使用。

```bash
# 使用Python推理
python deploy/python/det_keypoint_unite_infer.py \
          --det_model_dir output_inference/picodet_s_320 \
          --keypoint_model_dir output_inference/tinypose_256x192 \
          --video_file test_video.mp4 --device gpu --smooth True

# 使用CPP推理
./deploy/cpp/build/main --det_model_dir output_inference/picodet_s_320 \
          --keypoint_model_dir output_inference/tinypose_256x192 \
          --video_file test_video.mp4 --device gpu --smooth True
```
效果如下：

![](https://user-images.githubusercontent.com/15810355/181733125-3710bacc-2080-47e4-b397-3621a2f0caae.gif)

## BenchMark

我们给出了不同运行环境下的测试结果，供您在选用模型时参考。详细数据请见[Keypoint Inference Benchmark](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/keypoint/KeypointBenchmark.md)。

## 引用

```
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}

@InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
    title = {Distribution-Aware Coordinate Representation for Human Pose Estimation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```
