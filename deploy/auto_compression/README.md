# 自动化压缩

目录：
- [自动化压缩](#自动化压缩)
  - [1. 简介](#1-简介)
  - [2.Benchmark](#2benchmark)
    - [PP-YOLOE+](#pp-yoloe)
    - [YOLOv8](#yolov8)
    - [PP-YOLOE](#pp-yoloe-1)
    - [PP-PicoDet](#pp-picodet)
    - [RT-DETR](#rt-detr)
    - [DINO](#dino)
  - [3. 自动压缩流程](#3-自动压缩流程)
      - [3.1 准备环境](#31-准备环境)
      - [3.2 准备数据集](#32-准备数据集)
      - [3.3 准备预测模型](#33-准备预测模型)
      - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
      - [3.5 测试模型精度](#35-测试模型精度)
  - [4.预测部署](#4预测部署)
  - [5. FAQ:](#5-faq)
    - [1. ModuleNotFoundError: No module named 'ppdet'](#1-modulenotfounderror-no-module-named-ppdet)
    - [2. 精度问题排查](#2-精度问题排查)
    - [3. Pass排查脚本test\_pass.sh的使用](#3-pass排查脚本test_passsh的使用)

## 1. 简介
本示例使用PaddleDetection中Inference部署模型进行自动化压缩，使用的自动化压缩策略为量化蒸馏。


## 2.Benchmark


### PP-YOLOE+

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| PP-YOLOE+_s	 | 43.7  |  - | 42.9  |   -  |   -   |  -  |  [config](./configs/ppyoloe_plus_s_qat_dis.yaml) | [Quant Model](https://bj.bcebos.com/v1/paddledet/deploy/Inference/ppyoloe_plus_s_qat_dis.tar) |
| PP-YOLOE+_m | 49.8  |  - | 49.3  |   -  |   -   |  -  |  [config](./configs/ppyoloe_plus_m_qat_dis.yaml) | [Quant Model](https://bj.bcebos.com/v1/paddledet/deploy/Inference/ppyoloe_plus_m_qat_dis.tar) |
| PP-YOLOE+_l | 52.9  |  - | 52.6  |   -  |   -   |  -  |  [config](./configs/ppyoloe_plus_l_qat_dis.yaml) | [Quant Model](https://bj.bcebos.com/v1/paddledet/deploy/Inference/ppyoloe_plus_l_qat_dis.tar) |
| PP-YOLOE+_x | 54.7  |  - | 54.4  |   -  |   -   |  -  |  [config](./configs/ppyoloe_plus_x_qat_dis.yaml) | [Quant Model](https://bj.bcebos.com/v1/paddledet/deploy/Inference/ppyoloe_plus_x_qat_dis.tar) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

| 模型 | 策略 | TRT-mAP | GPU 耗时(ms) | MKLDNN-mAP | CPU 耗时(ms) | 配置文件 | 模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| PP-YOLOE+_crn_l_80e | Baseline |    52.88    |   12.4   |  52.88  |  522.6  |  [config](../../configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml)  | 待上传 |
| PP-YOLOE+_crn_l_80e | 量化蒸馏 | 52.52 | 7.2 | 52.65 | 539.5 | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/auto_compression/configs/ppyoloe_plus_l_qat_dis.yaml) | 待上传 |

- PP-YOLOE+_crn_l_80e mAP的指标在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- 上表测试环境：Tesla V100，Intel(R) Xeon(R) Gold 6271C，使用12线程测试，TensorRT 8.0.3.4，CUDA 11.2，Paddle2.5，batch_size=1。

### YOLOv8

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| YOLOv8-s | 44.9 |  43.9 | 44.3  |   9.27ms  |   4.65ms   |  **3.78ms**  |  [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/yolov8_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov8_s_500e_coco_trt_nms_quant.tar) |

**注意：**
- 表格中YOLOv8模型均为带NMS的模型，可直接在TRT中部署，如果需要对齐测试标准，需要测试不带NMS的模型。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- 表格中的性能在Tesla T4的GPU环境下测试，并且开启TensorRT，batch_size=1。

### PP-YOLOE

| 模型  | Base mAP | 离线量化mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 | TRT-INT8 |  配置文件 | 量化模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| PP-YOLOE-l | 50.9  |  - | 50.6  |   11.2ms  |   7.7ms   |  **6.7ms**  |  [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/auto_compression/configs/ppyoloe_l_qat_dis.yaml) | [Quant Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar) |
| PP-YOLOE-SOD | 38.5  |  - | 37.6  |   -  |   -   |  -  |  [config](./configs/ppyoloe_crn_l_80e_sliced_visdrone_640_025_qat.yml) | [Quant Model](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_sod_visdrone.tar) |

- PP-YOLOE-l mAP的指标在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- PP-YOLOE-l模型在Tesla V100的GPU环境下测试，并且开启TensorRT，batch_size=1，包含NMS，测试脚本是[benchmark demo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/python)。
- PP-YOLOE-SOD 的指标在VisDrone-DET数据集切图后的COCO格式[数据集](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone_sliced.zip)中评测得到，IoU=0.5:0.95。定义文件[ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml](../../configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml)

### PP-PicoDet

| 模型  | 策略 | mAP | FP32 | FP16 | INT8 |  配置文件 | 模型  |
| :-------- |:-------- |:--------: | :----------------: | :----------------: | :---------------: | :----------------------: | :---------------------: |
| PicoDet-S-NPU | Baseline | 30.1   |   -   |  -  |  -  | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_416_coco_npu.yml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar) |
| PicoDet-S-NPU |  量化训练 | 29.7  |   -  |   -   |  -  |  [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/full_quantization/detection/configs/picodet_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_npu_quant.tar) |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

|        模型        |   策略   | TRT-mAP | GPU 耗时(ms) | MKLDNN-mAP | ARM CPU 耗时(ms) |                           配置文件                           |                             模型                             |
| :----------------: | :------: | :-----: | :----------: | :--------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| PicoDet_s_320LCNet | Baseline |  29.06  |     3.6      |   29.06    |       42.0       | [config](../../configs/picodet/picodet_s_320_coco_lcnet.yml) | 待上传 |
| PicoDet_s_320LCNet | 量化蒸馏 |  28.82  |     3.3      |   28.58    |       46.7       | [config](./configs/picodet_s_320_lcnet_qat_dis.yaml) | 待上传 |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- 上表测试环境：Tesla V100，Intel(R) Xeon(R) Gold 6271C，使用12线程测试，TensorRT 8.0.3.4，CUDA 11.2，Paddle2.5，batch_size=1。

### RT-DETR

|       模型        | Base mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 |  TRT-INT8  |                           配置文件                           |                           量化模型                           |
| :---------------: | :------: | :--------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    RT-DETR-R50    |   53.1   |    53.0    | 32.05ms  |  9.12ms  | **6.96ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r50vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r50vd_6x_coco_quant.tar) |
|   RT-DETR-R101    |   54.3   |    54.1    | 54.13ms  | 12.68ms  | **9.20ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r101vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r101vd_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-L |   53.0   |    52.9    | 26.16ms  |  8.54ms  | **6.65ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_l_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-X |   54.8   |    54.6    | 49.22ms  | 12.50ms  | **9.24ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_x_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_x_6x_coco_quant.tar) |

- 上表测试环境：Tesla T4，TensorRT 8.6.0，CUDA 11.7，batch_size=1。

|       模型        | Base mAP | ACT量化mAP | TRT-FP32 | TRT-FP16 |  TRT-INT8  |                           配置文件                           |                           量化模型                           |
| :---------------: | :------: | :--------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    RT-DETR-R50    |   53.1   |    53.0    |  9.64ms  |  5.00ms  | **3.99ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r50vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r50vd_6x_coco_quant.tar) |
|   RT-DETR-R101    |   54.3   |    54.1    | 14.93ms  |  7.15ms  | **5.12ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_r101vd_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r101vd_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-L |   53.0   |    52.9    |  8.17ms  |  4.77ms  | **4.00ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_l_6x_coco_quant.tar) |
| RT-DETR-HGNetv2-X |   54.8   |    54.6    | 12.81ms  |  6.97ms  | **5.32ms** | [config](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_x_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_hgnetv2_x_6x_coco_quant.tar) |

- 上表测试环境：A10，TensorRT 8.6.0，CUDA 11.6，batch_size=1。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

|       模型        |   策略   | TRT-mAP | GPU 耗时(ms) | MKLDNN-mAP | ARM CPU 耗时(ms) |                           配置文件                           |                             模型                             |
| :---------------: | :------: | :-----: | :----------: | :--------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| RT-DETR-HGNetv2-L | Baseline |  53.09  |     32.7     |   52.54    |      3392.0      | [config](../../configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml) | 待上传 |
| RT-DETR-HGNetv2-L | 量化蒸馏 |  52.92  |     24.8     |   52.95    |      966.2       | [config](./configs/rtdetr_hgnetv2_l_qat_dis.yaml) | 待上传 |

- 上表测试环境：V100，Intel(R) Xeon(R) Gold 6271C，TensorRT 8.0.3.4，CUDA 11.2，Paddle2.5，batch_size=1。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

### DINO

| 模型                 |   策略   | TRT-mAP | GPU 耗时(ms) | MKLDNN-mAP | ARM CPU 耗时(ms) | 配置文件 |                             模型                             |
|:------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----:|:----:|
| DINO-R50-4scale-2x | Baseline |   50.82   |   147.7   |   待补充    |   待补充    | [config](../../configs/dino/dino_r50_4scale_2x_coco.yml)  | 待上传  |
| DINO-R50-4scale-2x | 量化蒸馏 |   50.72   |   127.9   |   待补充    |   待补充    |  [config](./configs/dino_r50_4scale_2x_qat_dis.yaml) | 待上传  |

- 上表测试环境：Tesla V100，TensorRT 8.0.3.4，Paddle2.5，CUDA 11.2，batch_size=1。
- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。
- cpu测试目前需要在paddle-develop版本下才能跑通。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.5 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.5.1
- PaddleDet >= 2.5
- opencv-python

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
pip install paddleslim
```

安装paddledet：
```shell
pip install paddledet
```

**注意：** YOLOv8模型的自动化压缩需要依赖安装最新[Develop Paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)和[Develop PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim#%E5%AE%89%E8%A3%85)版本。

#### 3.2 准备数据集

本案例默认以COCO数据进行自动压缩实验，如果自定义COCO数据，或者其他格式数据，请参考[数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/docs/tutorials/data/PrepareDataSet.md) 来准备数据。

如果数据集为非COCO格式数据，请修改[configs](./configs)中reader配置文件中的Dataset字段。

以PP-YOLOE模型为例，如果已经准备好数据集，请直接修改[./configs/yolo_reader.yml]中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。


根据[PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED_cn.md#8-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) 导出Inference模型，具体可参考下方PP-YOLOE模型的导出示例：
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型

PPYOLOE-l模型，包含NMS：如快速体验，可直接下载[PP-YOLOE-l导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar)
```shell
python tools/export_model.py \
        -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams \
        trt=True \
```

YOLOv8-s模型，包含NMS，具体可参考[YOLOv8模型文档](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov8), 然后执行：
```shell
python tools/export_model.py \
        -c configs/yolov8/yolov8_s_500e_coco.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams \
        trt=True
```

如快速体验，可直接下载[YOLOv8-s导出模型](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov8_s_500e_coco_trt_nms.tar)

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py \
        --act_config_path=./configs/ppyoloe_l_qat_dis.yaml \
        --config configs/ppyoloe_reader.yml \
        --save_dir='./output/' 
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --act_config_path=./configs/ppyoloe_l_qat_dis.yaml \
          --config configs/ppyoloe_reader.yml \
          --save_dir='./output/'
```

#### 3.5 测试模型精度

使用`test_det.py`脚本测试模型的mAP，以rtdetr为例子：
- 量化前+TensorRT
```
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco/ \
                --config ./configs/rtdetr_reader.yml \
                --precision fp32 \
                --use_trt True \
                --use_dynamic_shape False
```
- 量化后+TensorRT
```
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco_qat/ \
                --config ./configs/rtdetr_reader.yml \
                --precision int8 \
                --use_trt True \
                --use_dynamic_shape False
```
- 量化前+MKLDNN
```
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco/ \
                --config ./configs/rtdetr_reader.yml \
                --precision fp32 \
                --use_mkldnn True \
                --device CPU --cpu_threads 12 \
                --use_dynamic_shape False
```
- 量化后+MKLDNN
```
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco_qat/ \
                --config ./configs/rtdetr_reader.yml \
                --precision int8 \
                --use_mkldnn True \
                --device CPU --cpu_threads 12 \
                --use_dynamic_shape False
```
也可以使用`test_det.sh`脚本一键得到模型的mAP：
```
cd deploy/auto_compression
sh test_det.sh
```

**注意**：
- 在运行前请确保被测试模型的路径以及配置文件路径正确传入到model_path和config参数中。
- --precision 默认为paddle，如果使用trt，需要设置--use_trt=True，同时--precision 可设置为fp32/fp16/int8

## 4.预测部署

- 可以参考[PaddleDetection部署教程](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy)，GPU上量化模型开启TensorRT并设置trt_int8模式进行部署。

## 5. FAQ:

### 1. ModuleNotFoundError: No module named 'ppdet'

**A**: 运行：export PYTHONPATH=[path/to/ppdet]:$PYTHONPATH

### 2. 精度问题排查

**A**: 如果发现精度有问题，可以先参考官网的流程进行排查，参考[精度核验与问题追查](https://www.paddlepaddle.org.cn/inference/master/guides/performance_tuning/precision_tracing.html)，如果想确定是哪一个pass引起的精度问题，可以使用[test_pass.sh](./test_pass.sh)脚本进行测试，该脚本会将pass逐个删除然后进行测测试，通过查看输出日志便可以快速定位出现问题的pass。在运行脚本前请自行调整pass列表和测试选项。

### 3. Pass排查脚本[test_pass.sh](./test_pass.sh)的使用

**A**: 运行脚本前请先参考官网流程确认是pass出现问题。使用该脚本主要是为了找出哪个pass导致精度问题或者报错，脚本会逐个删除pass进行测试，通过查看输出日志便可以快速定位出现问题的pass。使用该脚本的步骤如下：
- 首先在终端日志中复制所有的pass名字，赋值给[test_pass.sh](./test_pass.sh)中的`pass_array`变量，配置输出日志文件的路径
- 然后调整脚本中运行`test_det.py`的参数，如模型路径、配置文件、精度等
- 最后即可运行脚本，通过查看输出日志文件便可以快速定位出现问题的pass
如果想一次排除两个个pass来测试，步骤如下：
- 在`test_det.py`中删掉想去掉的pass之一，接口为`config.delete_pass("pass_name")`
- 参考前面排除单个pass的步骤，运行脚本即可