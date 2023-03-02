简体中文

  <div align="center">
    <img src="https://user-images.githubusercontent.com/31800336/219260054-ba3766b1-8223-42bf-b69b-7092019995cc.jpg" width='600'/>
  </div>

# 3D Pose系列模型

## 目录

- [简介](#简介)
- [模型推荐](#模型推荐)
- [快速开始](#快速开始)
  - [环境安装](#1环境安装)
  - [数据准备](#2数据准备)
  - [训练与测试](#3训练与测试)
    - [单卡训练](#单卡训练)
    - [多卡训练](#多卡训练)
    - [模型评估](#模型评估)
    - [模型预测](#模型预测)
  - [使用说明](#4使用说明)

## 简介

PaddleDetection 中提供了两种3D Pose算法（稀疏关键点），分别是适用于服务器端的大模型Metro3D和移动端的TinyPose3D。其中Metro3D基于[End-to-End Human Pose and Mesh Reconstruction with Transformers](https://arxiv.org/abs/2012.09760)进行了稀疏化改造，TinyPose3D是在TinyPose基础上修改输出3D关键点。

## 模型推荐

|模型|适用场景|human3.6m精度(14关键点)|human3.6m精度(17关键点)|模型下载|
|:--:|:--:|:--:|:--:|:--:|
|Metro3D|服务器端|56.014|46.619|[metro3d_24kpts.pdparams](https://bj.bcebos.com/v1/paddledet/models/pose3d/metro3d_24kpts.pdparams)|
|TinyPose3D|移动端|86.381|71.223|[tinypose3d_human36m.pdparams](https://bj.bcebos.com/v1/paddledet/models/pose3d/tinypose3d_human36M.pdparams)|

注：
1. 训练数据基于 [MeshTransfomer](https://github.com/microsoft/MeshTransformer) 中的训练数据。
2. 测试精度同 MeshTransfomer 采用 14 关键点测试。

## 快速开始

### 1、环境安装

​    请参考PaddleDetection [安装文档](../../docs/tutorials/INSTALL_cn.md)正确安装PaddlePaddle和PaddleDetection即可。

### 2、数据准备

  我们的训练数据由coco、human3.6m、hr-lspet、posetrack3d、mpii组成。

​  2.1 我们的训练数据下载地址为：

  [coco](https://bj.bcebos.com/v1/paddledet/data/coco.tar)

  [human3.6m](https://bj.bcebos.com/v1/paddledet/data/pose3d/human3.6m.tar.gz)

  [lspet+posetrack+mpii](https://bj.bcebos.com/v1/paddledet/data/pose3d/pose3d_others.tar.gz)

  [标注文件下载](https://bj.bcebos.com/v1/paddledet/data/pose3d/pose3d.tar.gz)

  2.2 数据下载后按如下结构放在repo目录下

```
${REPO_DIR}  
|-- dataset  
|   |-- traindata
|       |-- coco
|       |-- hr-lspet
|       |-- human3.6m
|       |-- mpii
|       |-- posetrack3d
|       \-- pose3d
|           |-- COCO2014-All-ver01.json
|           |-- COCO2014-Part-ver01.json
|           |-- COCO2014-Val-ver10.json
|           |-- Human3.6m_train.json
|           |-- Human3.6m_valid.json
|           |-- LSPet_train_ver10.json
|           |-- LSPet_test_ver10.json
|           |-- MPII_ver01.json
|           |-- PoseTrack_ver01.json
|-- ppdet
|-- deploy
|-- demo
|-- README_cn.md
|-- README_en.md
|-- ...
```


### 3、训练与测试

#### 单卡训练

```shell
#单卡训练
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/pose3d/metro3d_24kpts.yml

#多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/pose3d/metro3d_24kpts.yml
```

#### 模型评估

```shell
#单卡评估
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/pose3d/metro3d_24kpts.yml -o weights=output/metro3d_24kpts/best_model.pdparams

#当只需要保存评估预测的结果时，可以通过设置save_prediction_only参数实现，评估预测结果默认保存在output/keypoints_results.json文件中
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/pose3d/metro3d_24kpts.yml -o weights=output/metro3d_24kpts/best_model.pdparams --save_prediction_only

#多卡评估
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/eval.py -c configs/pose3d/metro3d_24kpts.yml -o weights=output/metro3d_24kpts/best_model.pdparams
```

#### 模型预测

```shell
#图片生成3视角图
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/pose3d/metro3d_24kpts.yml -o weights=./output/metro3d_24kpts/best_model.pdparams --infer_img=./demo/hrnet_demo.jpg --draw_threshold=0.5
```

### 4、使用说明

  3D Pose在使用中相比2D Pose有更多的困难，该困难主要是由于以下两个原因导致的。

  - 1）训练数据标注成本高；

  - 2）图像在深度信息上的模糊性；

  由于（1）的原因训练数据往往只能覆盖少量动作，导致模型泛化性困难。由于（2）的原因图像在预测3D Pose坐标时深度z轴上误差通常大于x、y方向，容易导致时序间的较大抖动，且数据标注误差越大该问题表现的更加明显。

  要解决上述两个问题，就造成了两个矛盾的需求：1）提高泛化性需要更多的标注数据；2）降低预测误差需要高精度的数据标注。而3D Pose本身数据标注的困难导致越高精度的标注成本越高，标注数量则会相应降低。

  因此，我们提供的解决方案是：

  - 1）使用自动拟合标注方法自动产生大量低精度的数据。训练第一版模型，使其具有较普遍的泛化性。

  - 2）标注少量目标动作的高精度数据，基于第一版模型finetune，得到目标动作上的高精度模型，且一定程度上继承了第一版模型的泛化性。

  我们的训练数据提供了大量的低精度自动生成式的数据，用户可以在此数据训练的基础上，标注自己高精度的目标动作数据进行finetune，即可得到相对稳定较好的模型。

  我们在医疗康复高精度数据上的训练效果展示如下  [高清视频](https://user-images.githubusercontent.com/31800336/218949226-22e6ab25-facb-4cc6-8eca-38d4bfd973e5.mp4)

  <div align="center">
    <img src="https://user-images.githubusercontent.com/31800336/221747019-ceacfd64-e218-476b-a369-c6dc259816b2.gif" width='600'/>
  </div>



## 引用

```
@inproceedings{lin2021end-to-end,
author = {Lin, Kevin and Wang, Lijuan and Liu, Zicheng},
title = {End-to-End Human Pose and Mesh Reconstruction with Transformers},
booktitle = {CVPR},
year = {2021},
}
```
