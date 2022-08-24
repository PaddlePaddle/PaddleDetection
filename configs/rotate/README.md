简体中文 | [English](README_en.md)

# 旋转框检测

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [数据准备](#数据准备)
- [安装依赖](#安装依赖)

## 简介
旋转框常用于检测带有角度信息的矩形框，即矩形框的宽和高不再与图像坐标轴平行。相较于水平矩形框，旋转矩形框一般包括更少的背景信息。旋转框检测常用于遥感等场景中。

## 模型库

| 模型 | mAP | 学习率策略 | 角度表示 | 数据增广 | GPU数目 | 每GPU图片数目 | 模型下载 | 配置文件 |
|:---:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| [S2ANet](./s2anet/README.md) | 74.0 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/dota/s2anet_alignconv_2x_dota.yml) |

**注意:**

- 如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。

## 数据准备
### DOTA数据准备
DOTA数据集是一个大规模的遥感图像数据集，包含旋转框和水平框的标注。可以从[DOTA数据集官网](https://captain-whu.github.io/DOTA/)下载数据集并解压，解压后的数据集目录结构如下所示：
```
${DOTA_ROOT}
├── test
│   └── images
├── train
│   ├── images
│   └── labelTxt
└── val
    ├── images
    └── labelTxt
```

DOTA数据集分辨率较高，因此一般在训练和测试之前对图像进行切图，使用单尺度进行切图可以使用以下命令：
```
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval1024/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0
```
使用多尺度进行切图可以使用以下命令：
```
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 500 \
    --rates 0.5 1.0 1.5 \
```
对于无标注的数据可以设置`--image_only`进行切图，如下所示：
```
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/test/ \
    --output_dir ${OUTPUT_DIR}/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0 \
    --image_only
```

## 安装依赖
旋转框检测模型需要依赖外部算子进行训练，评估等。Linux环境下，你可以执行以下命令进行编译安装
```
cd ppdet/ext_op
python setup.py install
```
Windows环境请按照如下步骤安装：

（1）准备Visual Studio (版本需要>=Visual Studio 2015 update3)，这里以VS2017为例；

（2）点击开始-->Visual Studio 2017-->适用于 VS 2017 的x64本机工具命令提示；

（3）设置环境变量：`set DISTUTILS_USE_SDK=1`

（4）进入`PaddleDetection/ppdet/ext_op`目录，通过`python setup.py install`命令进行安装。

安装完成后，可以执行`ppdet/ext_op/unittest`下的单测验证外部op是否正确安装
