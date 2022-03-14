简体中文 | [English](README.md)

# PP-YOLOE

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [使用说明](#使用说明)
- [附录](#附录)

## 简介
PP-YOLOE是基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的yolo模型。PP-YOLOE有一系列的模型，即s/m/l/x，可以通过width multiplier和depth multiplier配置。PP-YOLOE避免使用诸如DCN或者matrix nms之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。更多细节可以参考我们的report。

PP-YOLOE-l在COCO test-dev2017达到了51.4的mAP, 同时其速度在Tesla V100上达到了78.1 FPS。PP-YOLOE-s/m/x同样具有卓越的精度速度性价比, 其精度速度可以在[模型库](#模型库)中找到。

PP-YOLOE由以下方法组成
- 可扩展的backbone和neck
- [Task Alignment Learning](https://arxiv.org/abs/2108.07755)
- Efficient Task-aligned head with [DFL](https://arxiv.org/abs/2006.04388)和[VFL](https://arxiv.org/abs/2008.13367)
- [SiLU激活函数](https://arxiv.org/abs/1710.05941)

## 模型库
|          模型           | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件  |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: | :------: |
| PP-YOLOE-s                  |     8      |     32     | cspresnet-s |     640     |       42.7        |        43.1         |      208.3      |          333.3          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m                  |     8      |     32     | cspresnet-m |     640     |       48.6        |        48.9         |      123.4      |          208.3          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l                  |     8      |     24     | cspresnet-l |     640     |       50.9        |        51.4         |      78.1      |          149.2          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x                  |     8      |     16     | cspresnet-x |     640     |       51.9        |        52.2         |      45.0      |          95.2          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)                   |

**注意:**

- PP-YOLOE模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集，Box AP<sup>test</sup>为`mAP(IoU=0.5:0.95)`评估结果。
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果训练GPU数和batch size不使用上述配置，须参考[FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/FAQ)调整学习率和迭代次数。
- PP-YOLOE模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.6.5，TensorRT推理速度测试使用TensorRT 6.0.1.8。
- PP-YOLOE推理速度测试使用`tools/export_model.py`脚本导出的模型，并用`deploy/python/infer.py`设置`--run_benchnark`参数得到。测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。
- 如果你设置了`--run_benchnark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

## 使用教程

### 1. 训练

执行以下指令使用混合精度训练PP-YOLOE

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --amp
```

### 2. 评估

执行以下命令在单个GPU上评估COCO val2017数据集

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并像`configs/ppyolo/ppyolo_test.yml`一样配置`EvalDataset`。

### 3. 推理

使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。


```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams --infer_dir=demo
```

### 4. 部署

PP-YOLOE在GPU上部署或者推理benchmark需要通过`tools/export_model.py`导出模型。

当你使用PaddleInferenced但不使用TensorRT时，运行以下的命令进行导出

```bash
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

当你使用PaddleInference的TensorRT时，需要指定`-o trt=True`进行导出

```bash
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams -o trt=True
```

`deploy/python/infer.py`使用上述导出后的PaddleInference模型用于推理和benchnark.

```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_dir=demo/ --device=gpu

# benchmark
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True
```

如果你想将PP-YOLOE模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)


```bash
# 导出推理模型
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/ppyoloe_crn_l_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file ppyoloe_crn_l_300e_coco.onnx

```

## 附录

PP-YOLOE消融实验

| 序号 |        模型                  | Box AP<sup>val</sup> | 参数量(M) | FLOPs(G) | V100 FP32 FPS |
| :--: | :---------------------------: | :-------------------: | :-------: | :------: | :-----------: |
|  A   | PP-YOLOv2          |         49.1         |   -   |  -   |      -     |
|  B   | A + Anchor-free    |         48.8         |   -   |  -   |      -     |
|  C   | B + CSP-ResNet     |         49.5         |   -   |  -   |      -     |
|  D   | C + TAL            |         50.4         |   -   |  -   |      -     |
|  E   | D + ET-Head        |         50.9         |   -   |  -   |      -     |
