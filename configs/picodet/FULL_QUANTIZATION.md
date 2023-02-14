# PP-PicoDet全量化示例

目录：

- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.全量化流程](#全量化流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 全精度模型训练](#33-全精度模型训练)
  - [3.4 导出预测模型](#33-导出预测模型)
  - [3.5 全量化并产出模型](#35-全量化并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介

本示例以PicoDet为例，介绍从模型训练、模型全量化，到NPU硬件上部署的全流程。

* [Benchmark](#Benchmark)表格中已经提供了基于COCO数据预训练模型全量化的模型。

* 已经验证的NPU硬件：

  - 瑞芯微-开发板：Rockchip RV1109、Rockchip RV1126、Rockchip RK1808

  - 晶晨-开发板：Amlogic A311D、Amlogic S905D3、Amlogic C308X

  - 恩智浦-开发板：NXP i.MX 8M Plus

 * 未验证硬件部署思路：
    - 未验证，表示该硬件暂不支持Paddle Lite推理部署，可以选择Paddle2ONNX导出，使用硬件的推理引擎完成部署，前提该硬件支持ONNX的全量化模型。

## 2.Benchmark

### PicoDet-S-NPU

| 模型            | 策略       | mAP  | FP32 | INT8 | 配置文件                                                                                                                              | 模型                                                                                  |
|:------------- |:-------- |:----:|:----:|:----:|:---------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|
| PicoDet-S-NPU | Baseline | 30.1 | -    | -    | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet/picodet_s_416_coco_npu.yml)                 | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar) |
| PicoDet-S-NPU | 量化训练     | 29.7 | -    | -    | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/full_quantization/detection/configs/picodet_s_qat_dis.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_npu_quant.tar)    |

- mAP的指标均在COCO val2017数据集中评测得到，IoU=0.5:0.95。

## 3. 全量化流程
基于自己数据训练的模型，可以参考如下流程。

### 3.1 准备环境

- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3
- PaddleDet >= 2.4

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

### 3.2 准备数据集

本案例默认以COCO数据进行全量化实验，如果自定义数据，可将数据按照COCO数据的标准准备；其他自定义数据，可以参考[PaddleDetection数据准备文档](../../docs/tutorials/data/PrepareDataSet.md) 来准备。

以PicoDet-S-NPU模型为例，如果已经准备好数据集，请直接修改[picodet_reader.yml](./configs/picodet_reader.yml)中`EvalDataset`的`dataset_dir`字段为自己数据集路径即可。

### 3.3 全精度模型训练

如需模型全量化，需要准备一个训好的全精度模型，如果已训好模型可跳过该步骤。

- 单卡GPU上训练:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/picodet/picodet_s_416_coco_npu.yml --eval
```

**注意：**如果训练时显存out memory，将TrainReader中batch_size调小，同时LearningRate中base_lr等比例减小。同时我们发布的config均由4卡训练得到，如果改变GPU卡数为1，那么base_lr需要减小4倍。

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/picodet/picodet_s_416_coco_npu.yml --eval
```

**注意：**PicoDet所有模型均由4卡GPU训练得到，如果改变训练GPU卡数，需要按线性比例缩放学习率base_lr。

- 评估:

```shell
python tools/eval.py -c configs/picodet/picodet_s_416_coco_npu.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_npu.pdparams
```

### 3.4 导出预测模型

使用如下命令，导出Inference模型，用于全量化训练。导出模型默认存放在`output_inference`文件夹,包括*.pdmodel和*.pdiparams文件，用于全量化。

* 命令说明：
    - -c: [3.3 全精度模型训练](#3.3全精度模型训练)训练时使用的yam配置文件。
    - -o weight: 预测模型文件，该文档直接使用基于COCO上训练好的模型。

```shell
python tools/export_model.py \
        -c configs/picodet/picodet_s_416_coco_npu.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_npu.pdparams \
```

### 3.5 全量化训练并产出模型

- 进入PaddleSlim自动化压缩Demo文件夹下：

  ```shell
  cd deploy/auto_compression/
  ```

全量化示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行全量化。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：

- 单卡量化训练：

  ```
  export CUDA_VISIBLE_DEVICES=0
  python run.py --config_path=./configs/picodet_s_qat_dis.yaml --save_dir='./output/'
  ```

- 多卡量化训练：

  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3
  python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/picodet_s_qat_dis.yaml --save_dir='./output/'
  ```

- 最终模型默认产出在`output`文件夹下，训练完成后，测试全量化模型精度

将config要测试的模型路径可以在配置文件中`model_dir`字段下进行修改。使用eval.py脚本得到模型的mAP：

```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/picodet_s_qat_dis.yaml
```

## 4.预测部署

请直接使用PicoDet的[Paddle Lite全量化Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)进行落地部署。

## 5.FAQ
