English | [简体中文](README.md)

# Rotated Object Detection

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model-Zoo)
- [Data Preparation](#Data-Preparation)
- [Installation](#Installation)

## Introduction
Rotated object detection is used to detect rectangular bounding boxes with angle information, that is, the long and short sides of the rectangular bounding box are no longer parallel to the image coordinate axes. Oriented bounding boxes generally contain less background information than horizontal bounding boxes. Rotated object detection is often used in remote sensing scenarios.

## Model Zoo
| Model | mAP | Lr Scheduler | Angle | Aug | GPU Number | images/GPU | download | config |
|:---:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| [S2ANet](./s2anet/README_en.md) | 73.84 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml) |
| [FCOSR](./fcosr/README_en.md) | 76.62 | 3x | oc | RR | 4 | 4 | [model](https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr/fcosr_x50_3x_dota.yml) |
| [PP-YOLOE-R-s](./ppyoloe_r/README_en.md) | 73.82 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota.yml) |
| [PP-YOLOE-R-s](./ppyoloe_r/README_en.md) | 79.42 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota_ms.yml) |
| [PP-YOLOE-R-m](./ppyoloe_r/README_en.md) | 77.64 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota.yml) |
| [PP-YOLOE-R-m](./ppyoloe_r/README_en.md) | 79.71 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota_ms.yml) |
| [PP-YOLOE-R-l](./ppyoloe_r/README_en.md) | 78.14 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml) |
| [PP-YOLOE-R-l](./ppyoloe_r/README_en.md) | 80.02 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_ms.yml) |
| [PP-YOLOE-R-x](./ppyoloe_r/README_en.md) | 78.28 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota.yml) |
| [PP-YOLOE-R-x](./ppyoloe_r/README_en.md) | 80.73 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota_ms.yml) |

**Notes:**

- if **GPU number** or **mini-batch size** is changed, **learning rate** should be adjusted according to the formula **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**.
- Models in model zoo is trained and tested with single scale by default. If `MS` is indicated in the data augmentation column, it means that multi-scale training and multi-scale testing are used. If `RR` is indicated in the data augmentation column, it means that RandomRotate data augmentation is used for training.

## Data Preparation
### DOTA Dataset preparation
The DOTA dataset is a large-scale remote sensing image dataset containing annotations of oriented and horizontal bounding boxes. The dataset can be download from [Official Website of DOTA Dataset](https://captain-whu.github.io/DOTA/). When the dataset is decompressed, its directory structure is shown as follows.
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

For labeled data, each image corresponds to a txt file with the same name, and each row in the txt file represent a rotated bouding box. The format is as follows:

```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

#### Slicing data with single scale
The image resolution of DOTA dataset is relatively high, so we usually slice the images before training and testing. To slice the images with a single scale, you can use the command below
``` bash
# slicing labeled data
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval1024/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0
# slicing unlabeled data by setting --image_only
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/test/ \
    --output_dir ${OUTPUT_DIR}/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0 \
    --image_only

```

#### Slicing data with multi scale
To slice the images with multiple scales, you can use the command below
``` bash
# slicing labeled data
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/train/ ${DOTA_ROOT}/val/ \
    --output_dir ${OUTPUT_DIR}/trainval/ \
    --coco_json_file DOTA_trainval1024.json \
    --subsize 1024 \
    --gap 500 \
    --rates 0.5 1.0 1.5
# slicing unlabeled data by setting --image_only
python configs/rotate/tools/prepare_data.py \
    --input_dirs ${DOTA_ROOT}/test/ \
    --output_dir ${OUTPUT_DIR}/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --subsize 1024 \
    --gap 500 \
    --rates 0.5 1.0 1.5 \
    --image_only
```

### Custom Dataset
Rotated object detction uses the standard COCO data format, and you can convert your dataset to COCO format to train the model. The annotations of standard COCO format contains the following information
``` python
'annotations': [
    {
        'id': 2083, 'category_id': 9, 'image_id': 9008,
        'bbox': [x, y, w, h], # horizontal bouding box
        'segmentation': [[x1, y1, x2, y2, x3, y3, x4, y4]], # rotated bounding box
        ...
    }
    ...
]
```
**It should be noted that `bbox` is the horizontal bouding box, and `segmentation` is four points of rotated bounding box (clockwise or counterclockwise). The `bbox` can be empty when training rotated object detector, and it is recommended to generate `bbox` according to `segmentation`**. In PaddleDetection 2.4 and earlier versions, `bbox` represents the rotated bounding box [x, y, w, h, angle] and `segmentation` is empty. **But this format is no longer supported after PaddleDetection 2.5, please download the latest dataset or convert to standard COCO format**.
## Installation
Models of rotated object detection depend on external operators for training, evaluation, etc. In Linux environment, you can execute the following command to compile and install.
```
cd ppdet/ext_op
python setup.py install
```
In Windows environment, perform the following steps to install it：

（1）Visual Studio (version required >= Visual Studio 2015 Update3);

（2）Go to Start --> Visual Studio 2017 --> X64 native Tools command prompt for VS 2017;

（3）Setting Environment Variables：set DISTUTILS_USE_SDK=1

（4）Enter `ppdet/ext_op` directory，use `python setup.py install` to install。

After the installation, you can execute the unittest of `ppdet/ext_op/unittest` to verify whether the external oprators is installed correctly.
