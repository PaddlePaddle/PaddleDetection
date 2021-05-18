# 模型压缩

在PaddleDetection中, 提供了基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)进行模型压缩的完整教程和benchmark。目前支持的方法：

- [剪裁](prune)
- [量化](quant)
- [蒸馏](distill)
- [联合策略](extensions)

推荐您使用剪裁和蒸馏联合训练，或者使用剪裁和量化，进行检测模型压缩。 下面以YOLOv3为例，进行剪裁、蒸馏和量化实验。

## 实验环境

- Python 3.7+
- PaddlePaddle >= 2.1.0
- PaddleSlim >= 2.0.0
- CUDA 10.1+
- cuDNN >=7.6.5

**PaddleDetection、 PaddlePaddle与PaddleSlim 版本关系:**
|  PaddleDetection版本  | PaddlePaddle版本  | PaddleSlim版本 |  备注    |
| :------------------: | :---------------: | :-------: |:---------------: |
| release/2.1       |       >= 2.1.0       | 2.1      |  --  |
| release/2.0       |       >= 2.0.1       | 2.0      | 量化依赖Paddle 2.1及PaddleSlim 2.1 |


#### 安装PaddleSlim
- 方法一：直接安装：
```
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 方法二：编译安装：
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```

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


## Benchmark

### 剪裁

#### Pascal VOC上benchmark

| 模型         |  压缩策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 | 预测时延（SD855）|   Box AP   |                           下载                          | 模型配置文件 | 压缩算法配置文件  |
| :----------------| :-------: | :------------: | :-------------: | :------: | :--------: | :------: | :-----------------------------------------------------: |:-------------: | :------: |
| YOLOv3-MobileNetV1      |  baseline | 24.13          |  93          |   608    | 289.9ms | 75.1       | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  -  |
| YOLOv3-MobileNetV1      |  剪裁-l1_norm(sensity) | 15.78(-34.49%) |  66(-29%) |   608   | - | 78.4(+3.3) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_voc_prune_l1_norm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/prune/yolov3_prune_l1_norm.yml)  |

- 目前剪裁支持YOLO系列、SSD、TTFNet、BlazeFace，其余模型正在开发支持中。
- SD855预测时延为使用PaddleLite部署，使用arm8架构并使用4线程(4 Threads)推理时延。

### 量化

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 |   Box AP    |                             下载                             |                         模型配置文件                         |                       压缩算法配置文件                       |
| ------------------ | ------------ | -------- | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline     | 608      |    28.8     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                            -                               |
| YOLOv3-MobileNetV1 | 普通在线量化 | 608      | 30.5 (+1.7) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_qat.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/quant/yolov3_mobilenet_v1_qat.yml) |
| YOLOv3-MobileNetV3 | baseline     | 608      |    31.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV3 | PACT在线量化 | 608      | 29.1 (-2.3) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_coco_qat.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/quant/yolov3_mobilenet_v3_qat.yml) |
| YOLOv3-DarkNet53 | baseline     | 608      |    39.0     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_darknet53_270e_coco.yml) |                           -                               |
| YOLOv3-DarkNet53 | 普通在线量化 | 608      | 38.8 (-0.2) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_darknet53_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/quant/yolov3_darknet_qat.yml) |
| SSD-MobileNet_v1    |  baseline   |   300   |  73.8  | [下载链接](https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) |     -    |
| SSD-MobileNet_v1    |  普通在线量化   |   300   |  72.9(-0.9)  | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_voc_qat.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/quant/ssd_mobilenet_v1_qat.yml) |
| Mask-ResNet50-FPN     |    baseline      |    (800, 1333)   |  39.2/35.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |  -  |
| Mask-ResNet50-FPN     |    普通在线量化      |    (800, 1333)   |  39.7(+0.5)/35.9(+0.3)    | [下载链接](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_qat.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/quant/mask_rcnn_r50_fpn_1x_qat.yml)  |


### 蒸馏

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 |   Box AP    |                             下载                             |                         模型配置文件                         |                       压缩算法配置文件                       |
| ------------------ | ------------ | -------- | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline     | 608      |    29.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV1 | 蒸馏 | 608      | 31.0(+1.6) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml) |

- 具体蒸馏方法请参考[蒸馏策略文档](distill/README.md)

### 蒸馏剪裁联合策略

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 | GFLOPs | 模型体积(MB) |  Box AP    |                             下载                             |                         模型配置文件                         |                       压缩算法配置文件                       |
| ------------------ | ------------ | -------- | :---------: |:---------: | :---------: |:----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline     | 608      | 24.65 | 94.6 |  29.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV1 | 蒸馏+剪裁 | 608      | 7.54(-69.4%) | 32.0(-66.0%) | 28.4(-1.0) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill_prune.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim/extensions/yolov3_mobilenet_v1_coco_distill_prune.yml) |
