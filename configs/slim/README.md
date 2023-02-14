# 模型压缩

在PaddleDetection中, 提供了基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)进行模型压缩的完整教程和benchmark。目前支持的方法：

- [剪裁](prune)
- [量化](quant)
- [离线量化](post_quant)
- [蒸馏](distill)
- [联合策略](extensions)

推荐您使用剪裁和蒸馏联合训练，或者使用剪裁、量化训练和离线量化，进行检测模型压缩。 下面以YOLOv3为例，进行剪裁、蒸馏和量化实验。

## 实验环境

- Python 3.7+
- PaddlePaddle >= 2.1.0
- PaddleSlim >= 2.1.0
- CUDA 10.1+
- cuDNN >=7.6.5

**PaddleDetection、 PaddlePaddle与PaddleSlim 版本关系:**
|  PaddleDetection版本  | PaddlePaddle版本  | PaddleSlim版本 |  备注    |
| :------------------: | :---------------: | :-------: |:---------------: |
| release/2.3       |       >= 2.1       | 2.1      | 离线量化依赖Paddle 2.2及PaddleSlim 2.2 |
| release/2.1 | 2.2    |       >= 2.1.0       | 2.1      |  量化模型导出依赖最新Paddle develop分支，可在[PaddlePaddle每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev)中下载安装  |
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


## 全链条部署

### 动转静导出模型

```shell
python tools/export_model.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} -o weights=output/{SLIM_CONFIG}/model_final
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。

### 部署预测

- Paddle-Inference预测：
    - [Python部署](../../deploy/python/README.md)
    - [C++部署](../../deploy/cpp/README.md)
    - [TensorRT预测部署教程](../../deploy/TENSOR_RT.md)
- 服务器端部署：使用[PaddleServing](../../deploy/serving/README.md)部署。
- 手机移动端部署：使用[Paddle-Lite](../../deploy/lite/README.md) 在手机移动端部署。

## Benchmark

### 剪裁

#### Pascal VOC上benchmark

| 模型         |  压缩策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 | 预测时延(SD855) |   Box AP   |                           下载                          | 模型配置文件 | 压缩算法配置文件  |
| :---------: | :-------: | :------------: |:-------------: | :------: | :-------------: | :------: | :-----------------------------------------------------: |:-------------: | :------: |
| YOLOv3-MobileNetV1      |  baseline | 24.13          |  93          |   608    | 332.0ms | 75.1       | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  -  |
| YOLOv3-MobileNetV1      |  剪裁-l1_norm(sensity) | 15.78(-34.49%) |  66(-29%) |   608   | - | 78.4(+3.3) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_voc_prune_l1_norm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/prune/yolov3_prune_l1_norm.yml)  |

#### COCO上benchmark
| 模型         |  压缩策略 |     GFLOPs     |  模型体积(MB)   | 输入尺寸 | 预测时延(SD855) |   Box AP   |                           下载                          | 模型配置文件 | 压缩算法配置文件  |
| :---------: | :-------: | :------------: |:-------------: | :------: | :-------------: | :------: | :-----------------------------------------------------: |:-------------: | :------: |
| PP-YOLO-MobileNetV3_large      |  baseline | -- | 18.5      |   608    | 25.1ms | 23.2       |  [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_mbv3_large_coco.yml)  |  -  |
| PP-YOLO-MobileNetV3_large      |  剪裁-FPGM | -37% | 12.6  |   608   | - | 22.3 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_prune_fpgm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_mbv3_large_coco.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/prune/ppyolo_mbv3_large_prune_fpgm.yml)  |
| YOLOv3-DarkNet53      |  baseline | -- | 238.2      |   608    | - | 39.0       |  [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_darknet53_270e_coco.yml)  |  -  |
| YOLOv3-DarkNet53      |  剪裁-FPGM | -24% | - |   608   | - | 37.6 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_prune_fpgm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_darknet53_270e_coco.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/prune/yolov3_darknet_prune_fpgm.yml)  |
| PP-YOLO_R50vd      |  baseline | -- | 183.3     |   608    | - | 44.8       |  [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)  |  -  |
| PP-YOLO_R50vd      |  剪裁-FPGM | -35% | -  |   608   | - | 42.1 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_prune_fpgm.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)  |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/prune/ppyolo_r50vd_prune_fpgm.yml)  |

说明：
- 目前剪裁除RCNN系列模型外，其余模型均已支持。
- SD855预测时延为使用PaddleLite部署，使用arm8架构并使用4线程(4 Threads)推理时延。

### 量化

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 |  模型体积(MB) | 预测时延(V100) | 预测时延(SD855) |  Box AP    |                  下载         |   Inference模型下载    |            模型配置文件                  |                     压缩算法配置文件                   |
| ------------------ | ------------ | -------- | :---------: | :---------: |:---------: | :---------: | :----------------------------------------------: | :----------------------------------------------: |:------------------------------------------: | :------------------------------------: |
| PP-YOLOE-l | baseline     | 640      | - |   11.2ms(trt_fp32) &#124; 7.7ms(trt_fp16)  | -- | 50.9     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | -  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | -  |
| PP-YOLOE-l | 普通在线量化     | 640      | - |   6.7ms(trt_int8)  | -- | 48.8     | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyoloe_l_coco_qat.pdparams) | -  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/ppyoloe_l_qat.yml) |
| PP-YOLOv2_R50vd | baseline     | 640      | 208.6  |   19.1ms  | -- | 49.1     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_365e_coco.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml) |                            -                               |
| PP-YOLOv2_R50vd | PACT在线量化     | 640      | --  |   17.3ms  | -- | 48.1     | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_qat.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml) |         [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/ppyolov2_r50vd_dcn_qat.yml)   |
| PP-YOLO_R50vd | baseline     | 608      | 183.3  |  17.4ms  | -- | 44.8     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_dcn_1x_coco.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml) |                  -         |
| PP-YOLO_R50vd | PACT在线量化     | 608      | 67.3  |  13.8ms  | -- | 44.3     | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_qat_pact.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_qat_pact.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml) |         [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/ppyolo_r50vd_qat_pact.yml)   |
| PP-YOLO-MobileNetV3_large | baseline     | 320    | 18.5   |  2.7ms  | 27.9ms | 23.2     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_coco.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_mbv3_large_coco.yml) |                            -                               |
| PP-YOLO-MobileNetV3_large | 普通在线量化     | 320   | 5.6   |  -- | 25.1ms | 24.3     | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_qat.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo/ppyolo_mbv3_large_coco.yml) |         [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/ppyolo_mbv3_large_qat.yml)   |
| YOLOv3-MobileNetV1 | baseline     | 608      | 94.2  | 8.9ms  |  332ms  |   29.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_270e_coco.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                            -                               |
| YOLOv3-MobileNetV1 | 普通在线量化 | 608      | 25.4  |  6.6ms  | 248ms  | 30.5 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_qat.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/yolov3_mobilenet_v1_qat.yml) |
| YOLOv3-MobileNetV3 | baseline     | 608    | 90.3  | 9.4ms  | 367.2ms  |    31.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_large_270e_coco.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV3 | PACT在线量化 | 608     | 24.4 |  8.0ms  |  280.0ms  | 31.1 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_coco_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_coco_qat.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/yolov3_mobilenet_v3_qat.yml) |
| YOLOv3-DarkNet53 | baseline     | 608    | 238.2  | 16.0ms | -- |  39.0     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet53_270e_coco.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_darknet53_270e_coco.yml) |                           -                               |
| YOLOv3-DarkNet53 | 普通在线量化 | 608      | 78.8  | 12.4ms | -- |  38.8 | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_darknet53_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/yolov3_darknet_qat.yml) |
| SSD-MobileNet_v1    |  baseline   |   300    | 22.5 | 4.4ms |  26.6ms |  73.8  | [下载链接](https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_120e_voc.tar) |[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) |     -    |
| SSD-MobileNet_v1    |  普通在线量化   |   300   | 7.1 | --  |  21.5ms | 72.9  | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_voc_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_voc_qat.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/ssd_mobilenet_v1_qat.yml) |
| Mask-ResNet50-FPN     |    baseline      |    (800, 1333)   | 174.1 | 359.5ms | -- |  39.2/35.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_coco.tar) |[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |  -  |
| Mask-ResNet50-FPN     |    普通在线量化    |    (800, 1333) | -- | -- | -- |  39.7(+0.5)/35.9(+0.3)    | [下载链接](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_qat.pdparams) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_qat.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |  [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/quant/mask_rcnn_r50_fpn_1x_qat.yml)  |

说明：
- 上述V100预测时延非量化模型均是使用TensorRT-FP32测试，量化模型均使用TensorRT-INT8测试，并且都包含NMS耗时。
- SD855预测时延为使用PaddleLite部署，使用arm8架构并使用4线程(4 Threads)推理时延。
- 上述PP-YOLOE模型均在V100，开启TensorRT环境中测速，不包含NMS。（导出模型时指定：-o trt=True exclude_nms=True）

### 离线量化
需要准备val集，用来对离线量化模型进行校准，运行方式：
```shell
python tools/post_quant.py -c configs/{MODEL.yml} --slim_config configs/slim/post_quant/{SLIM_CONFIG.yml}
```
例如：
```shell
python3.7 tools/post_quant.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml --slim_config=configs/slim/post_quant/ppyolo_mbv3_large_ptq.yml
```

### 蒸馏

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 |   Box AP    |                             下载                             |                         模型配置文件                         |                       压缩算法配置文件                       |
| ------------------ | ------------ | -------- | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline     | 608      |    29.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV1 | 蒸馏 | 608      | 31.0(+1.6) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml) |

- 具体蒸馏方法请参考[蒸馏策略文档](distill/README.md)

### 蒸馏剪裁联合策略

#### COCO上benchmark

| 模型               | 压缩策略     | 输入尺寸 | GFLOPs | 模型体积(MB) | 预测时延(SD855) |  Box AP    |                             下载                             |                         模型配置文件                         |                       压缩算法配置文件                       |
| ------------------ | ------------ | -------- | :---------: |:---------: |:---------: | :---------: |:----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline     | 608      | 24.65 | 94.2 | 332.0ms  |  29.4     | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                              -                               |
| YOLOv3-MobileNetV1 | 蒸馏+剪裁 | 608      | 7.54(-69.4%) | 30.9(-67.2%) | 166.1ms  | 28.4(-1.0) | [下载链接](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill_prune.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/extensions/yolov3_mobilenet_v1_coco_distill_prune.yml) |
| YOLOv3-MobileNetV1 | 剪裁+量化 | 608      | - | - | - | - | - | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) | [slim配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/slim/extensions/yolov3_mobilenetv1_prune_qat.yml) |
