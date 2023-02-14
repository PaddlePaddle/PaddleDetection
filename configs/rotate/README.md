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
| [S2ANet](./s2anet/README.md) | 73.84 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml) |
| [FCOSR](./fcosr/README.md) | 76.62 | 3x | oc | RR | 4 | 4 | [model](https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr/fcosr_x50_3x_dota.yml) |
| [PP-YOLOE-R-s](./ppyoloe_r/README.md) | 73.82 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota.yml) |
| [PP-YOLOE-R-s](./ppyoloe_r/README.md) | 79.42 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota_ms.yml) |
| [PP-YOLOE-R-m](./ppyoloe_r/README.md) | 77.64 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota.yml) |
| [PP-YOLOE-R-m](./ppyoloe_r/README.md) | 79.71 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota_ms.yml) |
| [PP-YOLOE-R-l](./ppyoloe_r/README.md) | 78.14 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml) |
| [PP-YOLOE-R-l](./ppyoloe_r/README.md) | 80.02 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_ms.yml) |
| [PP-YOLOE-R-x](./ppyoloe_r/README.md) | 78.28 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota.yml) |
| [PP-YOLOE-R-x](./ppyoloe_r/README.md) | 80.73 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota_ms.yml) |

**注意:**

- 如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 模型库中的模型默认使用单尺度训练单尺度测试。如果数据增广一栏标明MS，意味着使用多尺度训练和多尺度测试。如果数据增广一栏标明RR，意味着使用RandomRotate数据增广进行训练。

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

对于有标注的数据，每一张图片会对应一个同名的txt文件，文件中每一行为一个旋转框的标注，其格式如下：
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

#### 单尺度切图
DOTA数据集分辨率较高，因此一般在训练和测试之前对图像进行离线切图，使用单尺度进行切图可以使用以下命令：
``` bash
# 对于有标注的数据进行切图
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval1024/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0

# 对于无标注的数据进行切图需要设置--image_only
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/test/ \
    --output_dir ${OUTPUT_DIR}/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0 \
    --image_only

```

#### 多尺度切图
使用多尺度进行切图可以使用以下命令：
``` bash
# 对于有标注的数据进行切图
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 500 \
    --rates 0.5 1.0 1.5

# 对于无标注的数据进行切图需要设置--image_only
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/test/ \
    --output_dir ${OUTPUT_DIR}/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --subsize 1024 \
    --gap 500 \
    --rates 0.5 1.0 1.5 \
    --image_only
```

### 自定义数据集
旋转框使用标准COCO数据格式，你可以将你的数据集转换成COCO格式以训练模型。COCO标准数据格式的标注信息中包含以下信息：
``` python
'annotations': [
    {
        'id': 2083, 'category_id': 9, 'image_id': 9008,
        'bbox': [x, y, w, h], # 水平框标注
        'segmentation': [[x1, y1, x2, y2, x3, y3, x4, y4]], # 旋转框标注
        ...
    }
    ...
]
```
**需要注意的是`bbox`的标注是水平框标注，`segmentation`为旋转框四个点的标注(顺时针或逆时针均可)。在旋转框训练时`bbox`是可以缺省，一般推荐根据旋转框标注`segmentation`生成。** 在PaddleDetection 2.4及之前的版本，`bbox`为旋转框标注[x, y, w, h, angle]，`segmentation`缺省，**目前该格式已不再支持，请下载最新数据集或者转换成标准COCO格式**。

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
