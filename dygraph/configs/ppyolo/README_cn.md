简体中文 | [English](README.md)

# PP-YOLO 模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [使用说明](#使用说明)
- [未来工作](#未来工作)
- [附录](#附录)

## 简介

[PP-YOLO](https://arxiv.org/abs/2007.12099)是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)模型，要求使用PaddlePaddle 2.0.0rc1(可使用pip安装) 或适当的[develop版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release)。

PP-YOLO在[COCO](http://cocodataset.org) test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。

<div align="center">
  <img src="../../../docs/images/ppyolo_map_fps.png" width=500 />
</div>

PP-YOLO从如下方面优化和提升YOLOv3模型的精度和速度：

- 更优的骨干网络: ResNet50vd-DCN
- 更大的训练batch size: 8 GPUs，每GPU batch_size=24，对应调整学习率和迭代轮数
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- 更优的预训练模型

## 模型库

### PP-YOLO模型

|          模型            | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件 |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: | :------: |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     608     |         44.8         |         45.2          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     512     |         43.9         |         44.4          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     416     |         42.1         |         42.5          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     320     |         38.9         |         39.3          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     608     |         45.3         |         45.9          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     512     |         44.4         |         45.0          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     416     |         42.7         |         43.2          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     320     |         39.5         |         40.1          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_ResNet18vd               |     4      |     32     | ResNet18vd |     512     |         29.2         |         29.5          |      357.1      |          657.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO_ResNet18vd               |     4      |     32     | ResNet18vd |     416     |         28.6         |         28.9          |      409.8      |          719.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO_ResNet18vd               |     4      |     32     | ResNet18vd |     320     |         26.2         |         26.4          |      480.7      |          763.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |

**注意:**

- PP-YOLO模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集，Box AP<sup>test</sup>为`mAP(IoU=0.5:0.95)`评估结果。
- PP-YOLO模型训练过程中使用8 GPUs，每GPU batch size为24进行训练，如训练GPU数和batch size不使用上述配置，须参考[FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/FAQ.md)调整学习率和迭代次数。
- PP-YOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.5.1，TensorRT推理速度测试使用TensorRT 5.1.2.2。
- PP-YOLO模型FP32的推理速度测试数据为使用`tools/export_model.py`脚本导出模型后，使用`deploy/python/infer.py`脚本中的`--run_benchnark`参数使用Paddle预测库进行推理速度benchmark测试结果, 且测试的均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。
- TensorRT FP16的速度测试相比于FP32去除了`yolo_box`(bbox解码)部分耗时，即不包含数据预处理，bbox解码和NMS(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。
- PP-YOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.5.1，TensorRT推理速度测试使用TensorRT 5.1.2.2。

### PP-YOLO 轻量级模型

|          模型                | GPU个数 | 每GPU图片个数 |  模型体积  | 输入尺寸 | Box AP<sup>val</sup> |  Box AP50<sup>val</sup> | Kirin 990 1xCore (FPS) | 模型下载 |  配置文件 |
|:----------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: |  :--------------------: | :--------------------: | :------: | :------: |
| PP-YOLO_MobileNetV3_large    |    4    |      32       |    28MB    |   320    |         23.2         |           42.6          |           14.1         | [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_mbv3_large_coco.yml)                   |
| PP-YOLO_MobileNetV3_small    |    4    |      32       |    16MB    |   320    |         17.2         |           33.8          |           21.5         | [下载链接](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_small_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_mbv3_small_coco.yml)                   |

- PP-YOLO_MobileNetV3 模型使用COCO数据集中train2017作为训练集，使用val2017作为测试集，Box AP<sup>val</sup>为`mAP(IoU=0.5:0.95)`评估结果, Box AP50<sup>val</sup>为`mAP(IoU=0.5)`评估结果。
- PP-YOLO_MobileNetV3 模型训练过程中使用4GPU，每GPU batch size为32进行训练，如训练GPU数和batch size不使用上述配置，须参考[FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/FAQ.md)调整学习率和迭代次数。
- PP-YOLO_MobileNetV3 模型推理速度测试环境配置为麒麟990芯片单线程。

### Pascal VOC数据集上的PP-YOLO

PP-YOLO在Pascal VOC数据集上训练模型如下:

|       模型         | GPU个数 | 每GPU图片个数 |  骨干网络  |   输入尺寸  | Box AP50<sup>val</sup> | 模型下载 | 配置文件 |
|:------------------:|:-------:|:-------------:|:----------:| :----------:| :--------------------: | :------: | :-----: |
| PP-YOLO            |    8    |       12      | ResNet50vd |     608     |          84.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |
| PP-YOLO            |    8    |       12      | ResNet50vd |     416     |          84.3          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |
| PP-YOLO            |    8    |       12      | ResNet50vd |     320     |          82.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |

## 使用说明

### 1. 训练

使用8GPU通过如下命令一键式启动训练(以下命令均默认在PaddleDetection根目录运行), 通过`--eval`参数开启训练中交替评估。

```bash
python -m paddle.distributed.launch --log_dir=./ppyolo_dygraph/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml &>ppyolo_dygraph.log 2>&1 &
```

### 2. 评估

使用单GPU通过如下命令一键式评估模型在COCO val2017数据集效果

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=output/ppyolo_r50vd_dcn_1x_coco/model_final
```

我们提供了`configs/ppyolo/ppyolo_test.yml`用于评估COCO test-dev2017数据集的效果，评估COCO test-dev2017数据集的效果须先从[COCO数据集下载页](https://cocodataset.org/#download)下载test-dev2017数据集，解压到`configs/ppyolo/ppyolo_test.yml`中`EvalReader.dataset`中配置的路径，并使用如下命令进行评估

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_test.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_test.yml -o weights=output/ppyolo_r50vd_dcn_1x_coco/model_final
```

评估结果保存于`bbox.json`中，将其压缩为zip包后通过[COCO数据集评估页](https://competitions.codalab.org/competitions/20794#participate)提交评估。

**注意:** `configs/ppyolo/ppyolo_test.yml`仅用于评估COCO test-dev数据集，不用于训练和评估COCO val2017数据集。

### 3. 推理

使用单GPU通过如下命令一键式推理图像，通过`--infer_img`指定图像路径，或通过`--infer_dir`指定目录并推理目录下所有图像

```bash
# 推理单张图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=../demo/000000014439_640x640.jpg

# 推理目录下所有图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_dir=../demo
```

### 4. 推理部署

PP-YOLO模型部署及推理benchmark需要通过`tools/export_model.py`导出模型后使用Paddle预测库进行部署和推理，可通过如下命令一键式启动。

```bash
# 导出模型，默认存储于output/ppyolo目录
python tools/export_model.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# 预测库推理
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=../demo/000000014439_640x640.jpg --use_gpu=True
```


## 未来工作

1. 发布PP-YOLO-tiny模型
2. 发布更多骨干网络的PP-YOLO模型

## 附录

PP-YOLO模型相对于YOLOv3模型优化项消融实验数据如下表所示。

| 序号 |        模型                  | Box AP<sup>val</sup> | Box AP<sup>test</sup> | 参数量(M) | FLOPs(G) | V100 FP32 FPS |
| :--: | :--------------------------- | :------------------: | :-------------------: | :-------: | :------: | :-----------: |
|  A   | YOLOv3-DarkNet53             |         38.9         |            -          |   59.13   |  65.52   |      58.2     |
|  B   | YOLOv3-ResNet50vd-DCN        |         39.1         |            -          |   43.89   |  44.71   |      79.2     |
|  C   | B + LB + EMA + DropBlock     |         41.4         |            -          |   43.89   |  44.71   |      79.2     |
|  D   | C + IoU Loss                 |         41.9         |            -          |   43.89   |  44.71   |      79.2     |
|  E   | D + IoU Aware                |         42.5         |            -          |   43.90   |  44.71   |      74.9     |
|  F   | E + Grid Sensitive           |         42.8         |            -          |   43.90   |  44.71   |      74.8     |
|  G   | F + Matrix NMS               |         43.5         |            -          |   43.90   |  44.71   |      74.8     |
|  H   | G + CoordConv                |         44.0         |            -          |   43.93   |  44.76   |      74.1     |
|  I   | H + SPP                      |         44.3         |          45.2         |   44.93   |  45.12   |      72.9     |
|  J   | I + Better ImageNet Pretrain |         44.8         |          45.2         |   44.93   |  45.12   |      72.9     |
|  K   | J + 2x Scheduler             |         45.3         |          45.9         |   44.93   |  45.12   |      72.9     |

**注意:**

- 精度与推理速度数据均为使用输入图像尺寸为608的测试结果
- Box AP为在COCO train2017数据集训练，val2017和test-dev2017数据集上评估`mAP(IoU=0.5:0.95)`数据
- 推理速度为单卡V100上，batch size=1, 使用上述benchmark测试方法的测试结果，测试环境配置为CUDA 10.2，CUDNN 7.5.1
- [YOLOv3-DarkNet53](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_darknet53_270e_coco.yml)精度38.9为PaddleDetection优化后的YOLOv3模型，可参见[模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/MODEL_ZOO.md)
