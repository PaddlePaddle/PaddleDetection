# PPYOLO 模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [使用说明](#使用说明)
- [未来工作](#未来工作)
- [如何贡献代码](#如何贡献代码)

## 简介

[PPYOLO]()的PaddleDetection中基于YOLOv3模型优化的模型，其COCO数据集mAP和预测速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)模型，要求使用PaddlePaddle1.8.3或适当的develop版本。

PPYOLO在[COCO](http://cocodataset.org)val数据集上精度达到45.3%，在单卡V100上FP32预测速度为70.1 FPS, V100上开启TensorRT下FP16预测速度为129.8 FPS。

<div align="center">
  <img src="../../docs/images/ppyolo_map_fps.png" />
</div>

PPYOLO从如下方面优化和提升YOLOv3模型的精度和速度：

- 更优的骨干网络: ResNet50vd-DCN
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- 更大的训练batch size
- 更优的预训练模型

## 模型库

|          模型            | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP | V100 FP32(fps) | V100 TensorRT FP16(fps) | 模型下载 | 配置文件 |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :----: | :------------: | :---------------------: | :------: | :------: |
| PPYOLO                   |    8    |      24       | ResNet50vd |   608    |  45.3  |      70.1      |          129.9          | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ppyolo/ppyolo.yml)                   |
| PPYOLO                   |    8    |      24       | ResNet50vd |   512    |  44.4  |      89.9      |          153.4          | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ppyolo/ppyolo.yml)                   |
| PPYOLO                   |    8    |      24       | ResNet50vd |   416    |  42.5  |     109.1      |          176.1          | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ppyolo/ppyolo.yml)                   |
| PPYOLO                   |    8    |      24       | ResNet50vd |   320    |  39.3  |     132.2      |          202.2          | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/ppyolo/ppyolo.yml)                   |

**注意:**

- PPYOLO模型使用COCO数据集中train2017作为训练集，使用val2017左右测试集。
- PPYOLO模型训练过程中使用8GPU，每GPU batch size为24进行训练，如训练GPU数和batch size不使用上述配置，须参考[FAQ](../../docs/FAQ.md)调整学习率和迭代次数。
- PPYOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.0，TensorRT推理速度测试使用TensorRT 5.1.2.2。
- PPYOLO模型推理速度测试数据为使用`tools/export_model.py`脚本导出模型后，使用`deploy/python/infer.py`脚本中的`--run_benchnark`参数使用Paddle预测库进行推理速度benchmark测试结果, 且测试的均为不包含数据预处理和模型输出后处理(NMS)的数据(与YOLOv4测试方法一致)。

## 使用说明

### 1. 训练

使用8GPU通过如下命令一键式启动训练(以下命令均默认在PaddleDetection根目录运行), 通过`--eval`参数开启训练中交替评估。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train.py -c configs/ppyolo/ppyolo.yml --eval
```

### 2. 评估

使用单GPU通过如下命令一键式评估模型效果

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo.yml -o weights=output/ppyolo/best_model
```

### 3. 推理

使用单GPU通过如下命令一键式推理图像，通过`--infer_img`指定图像路径，或通过`--infer_dir`指定目录并推理目录下所有图像

```bash
# 推理单张图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理目录下所有图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_dir=demo
```

### 4. 推理部署与benchmark

PPYOLO模型部署及推理benchmark需要通过`tools/export_model.py`导出模型后使用Paddle预测库进行部署和推理，可通过如下命令一键式启动。

```bash
# 导出模型，默认存储于output/ppyolo目录
python tools/export_model.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams

# 预测库推理
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output/ppyolo --image_file=demo/000000014439_640x640.jpg --use_gpu=True
```

PPYOLO模型benchmark测试为不包含数据预处理和网络输出后处理(NMS)的网络结构部分数据，导出模型时须指定`--exlcude_nms`来裁剪掉模型中后处理的NMS部分，通过如下命令进行模型导出和benchmark测试。

```bash
# 导出模型，通过--exclude_nms参数裁剪掉模型中的NMS部分，默认存储于output/ppyolo目录
python tools/export_model.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --exclude_nms

# FP32 benchmark测试
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output/ppyolo --image_file=demo/000000014439_640x640.jpg --use_gpu=True --run_benchmark=True

# TensorRT FP16 benchmark测试
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output/ppyolo --image_file=demo/000000014439_640x640.jpg --use_gpu=True --run_benchmark=True --run_mode=trt_fp16
```

## 未来工作

1. 发布PPYOLO-tiny模型
2. 发布更多骨干网络的PPYOLO及PPYOLO-tiny模型

## 如何贡献代码
我们非常欢迎您可以为PaddleDetection提供代码，您可以提交PR供我们review；也十分感谢您的反馈，可以提交相应issue，我们会及时解答。
