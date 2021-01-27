# 模型压缩

在PaddleDetection中, 提供了基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)进行模型压缩的完整教程和benchmark。目前支持的方法：

- [剪裁](prune)

推荐您使用剪裁和蒸馏联合训练，或者使用剪裁和量化，进行检测模型压缩。 下面以YOLOv3为例，进行剪裁、蒸馏和量化实验。

## Benchmark

### 剪裁

#### Pascal VOC上benchmark

| 模型         |  压缩策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 | 预测时延（SD855）|   Box AP   |                           下载                          | 模型配置文件 | 压缩算法配置文件  |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------: | :------: | :-----------------------------------------------------: |:-------------: | :------: |
| YOLOv3-MobileNetV1      |  baseline | 24.13          |  93          |   608    | 289.9ms | 75.1       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  -  |
| YOLOv3-MobileNetV1      |  剪裁-l1_norm(sensity) | 15.78(-34.49%) |  66(-29%) |   608   | - | 77.6(+2.5) | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/slim/yolov3_mobilenet_v1_voc_prune_l1_norm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/slim/prune/yolov3_prune_l1_norm.yml)  |

- SD855预测时延为使用PaddleLite部署，使用arm8架构并使用4线程(4 Threads)推理时延

## 实验环境

- Python 3.7+
- PaddlePaddle >= 2.0.0
- PaddleSlim >= 2.0.0
- CUDA 9.0+
- cuDNN >=7.5

## 快速开始

### 训练

```shell
python tools/train.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml}
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。


### 评估

```shell
python tools/eval.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} -o weights=output/{SLIM_CONFIG}/model_final
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。

### 测试

```shell
python tools/infer.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} \
    -o weights=output/{SLIM_CONFIG}/model_final
    --infer_img={IMAGE_PATH}
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。
- `--infer_img`: 指定测试图像路径。


### 动转静导出模型

```shell
python tools/export_model.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} -o weights=output/{SLIM_CONFIG}/model_final
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。
