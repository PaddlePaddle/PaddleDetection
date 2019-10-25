>运行该示例前请安装Paddle1.6或更高版本

# 检测模型卷积通道剪裁示例

## 概述

该示例使用PaddleSlim提供的[卷积通道剪裁压缩策略](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md#2-%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%89%AA%E8%A3%81%E5%8E%9F%E7%90%86)对检测库中的模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- <a href="../../README_cn.md">检测库的常规训练方法</a>
- [检测模型数据准备](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/PaddleDetection/docs/INSTALL_cn.md#%E6%95%B0%E6%8D%AE%E9%9B%86)
- [PaddleSlim使用文档](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md)


## 配置文件说明

关于配置文件如何编写您可以参考：

- [PaddleSlim配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#122-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E7%9A%84%E4%BD%BF%E7%94%A8)
- [裁剪策略配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#22-%E6%A8%A1%E5%9E%8B%E9%80%9A%E9%81%93%E5%89%AA%E8%A3%81)

其中，配置文件中的`pruned_params`需要根据当前模型的网络结构特点设置，它用来指定要裁剪的parameters.

这里以MobileNetV1-YoloV3模型为例，其卷积可以三种：主干网络中的普通卷积，主干网络中的`depthwise convolution`和`yolo block`里的普通卷积。PaddleSlim暂时无法对`depthwise convolution`直接进行剪裁， 因为`depthwise convolution`的`channel`的变化会同时影响到前后的卷积层。我们这里只对主干网络中的普通卷积和`yolo block`里的普通卷积做裁剪。

通过以下方式可视化模型结构：

```
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

graph = IrGraph(core.Graph(train_prog.desc), for_test=True)
marked_nodes = set()
for op in graph.all_op_nodes():
    print(op.name())
    if op.name().find('conv') > -1:
        marked_nodes.add(op)
graph.draw('.', 'forward', marked_nodes)
```

该示例中MobileNetV1-YoloV3模型结构的可视化结果：<a href="./images/MobileNetV1-YoloV3.pdf">MobileNetV1-YoloV3.pdf</a>

同时通过以下命令观察目标卷积层的参数（parameters）的名称和shape:

```
for param in fluid.default_main_program().global_block().all_parameters():
    if 'weights' in param.name:
        print(param.name, param.shape)
```


从可视化结果，我们可以排除后续会做concat的卷积层，最终得到如下要裁剪的参数名称：

```
conv2_1_sep_weights
conv2_2_sep_weights
conv3_1_sep_weights
conv4_1_sep_weights
conv5_1_sep_weights
conv5_2_sep_weights
conv5_3_sep_weights
conv5_4_sep_weights
conv5_5_sep_weights
conv5_6_sep_weights
yolo_block.0.0.0.conv.weights
yolo_block.0.0.1.conv.weights
yolo_block.0.1.0.conv.weights
yolo_block.0.1.1.conv.weights
yolo_block.1.0.0.conv.weights
yolo_block.1.0.1.conv.weights
yolo_block.1.1.0.conv.weights
yolo_block.1.1.1.conv.weights
yolo_block.1.2.conv.weights
yolo_block.2.0.0.conv.weights
yolo_block.2.0.1.conv.weights
yolo_block.2.1.1.conv.weights
yolo_block.2.2.conv.weights
yolo_block.2.tip.conv.weights
```

```
(conv2_1_sep_weights)|(conv2_2_sep_weights)|(conv3_1_sep_weights)|(conv4_1_sep_weights)|(conv5_1_sep_weights)|(conv5_2_sep_weights)|(conv5_3_sep_weights)|(conv5_4_sep_weights)|(conv5_5_sep_weights)|(conv5_6_sep_weights)|(yolo_block.0.0.0.conv.weights)|(yolo_block.0.0.1.conv.weights)|(yolo_block.0.1.0.conv.weights)|(yolo_block.0.1.1.conv.weights)|(yolo_block.1.0.0.conv.weights)|(yolo_block.1.0.1.conv.weights)|(yolo_block.1.1.0.conv.weights)|(yolo_block.1.1.1.conv.weights)|(yolo_block.1.2.conv.weights)|(yolo_block.2.0.0.conv.weights)|(yolo_block.2.0.1.conv.weights)|(yolo_block.2.1.1.conv.weights)|(yolo_block.2.2.conv.weights)|(yolo_block.2.tip.conv.weights)
```

综上，我们将MobileNetV2配置文件中的`pruned_params`设置为以下正则表达式：

```
(conv2_1_sep_weights)|(conv2_2_sep_weights)|(conv3_1_sep_weights)|(conv4_1_sep_weights)|(conv5_1_sep_weights)|(conv5_2_sep_weights)|(conv5_3_sep_weights)|(conv5_4_sep_weights)|(conv5_5_sep_weights)|(conv5_6_sep_weights)|(yolo_block.0.0.0.conv.weights)|(yolo_block.0.0.1.conv.weights)|(yolo_block.0.1.0.conv.weights)|(yolo_block.0.1.1.conv.weights)|(yolo_block.1.0.0.conv.weights)|(yolo_block.1.0.1.conv.weights)|(yolo_block.1.1.0.conv.weights)|(yolo_block.1.1.1.conv.weights)|(yolo_block.1.2.conv.weights)|(yolo_block.2.0.0.conv.weights)|(yolo_block.2.0.1.conv.weights)|(yolo_block.2.1.1.conv.weights)|(yolo_block.2.2.conv.weights)|(yolo_block.2.tip.conv.weights)
```

我们可以用上述操作观察其它检测模型的参数名称规律，然后设置合适的正则表达式来剪裁合适的参数。

## 训练

根据<a href="../../tools/train.py">PaddleDetection/tools/train.py</a>编写压缩脚本compress.py。
在该脚本中定义了Compressor对象，用于执行压缩任务。

### 执行示例

step1: 设置gpu卡
```
export CUDA_VISIBLE_DEVICES=0
```
step2: 开始训练

使用PaddleDetection提供的配置文件在用8卡进行训练：

```
python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -o max_iters=258 \
    YoloTrainFeed.batch_size=64 \
    -d "../../dataset/voc"
```

>通过命令行覆盖设置max_iters选项，因为PaddleDetection中训练是以`batch`为单位迭代的，并没有涉及`epoch`的概念，但是PaddleSlim需要知道当前训练进行到第几个`epoch`, 所以需要将`max_iters`设置为一个`epoch`内的`batch`的数量。

如果要调整训练卡数，需要调整配置文件`yolov3_mobilenet_v1_voc.yml`中的以下参数：

- **max_iters:** 一个`epoch`中batch的数量，需要设置为`total_num / batch_size`, 其中`total_num`为训练样本总数量，`batch_size`为多卡上总的batch size.
- **YoloTrainFeed.batch_size:** 当使用DataLoader时，表示单张卡上的batch size; 当使用普通reader时，则表示多卡上的总的`batch_size`。`batch_size`受限于显存大小。
- **LeaningRate.base_lr:** 根据多卡的总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。
- **LearningRate.schedulers.PiecewiseDecay.LinearWarmup.steps：** 请根据batch size的变化对其进行调整。


以下为4卡训练示例，通过命令行覆盖`yolov3_mobilenet_v1_voc.yml`中的参数：

```
python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -o max_iters=258 \
    YoloTrainFeed.batch_size=64 \
    -d "../../dataset/voc"
```

以下为2卡训练示例，受显存所制，单卡`batch_size`不变，总`batch_size`减小，`base_lr`减小，一个epoch内batch数量增加，同时需要调整学习率相关参数，如下：
```
python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -o max_iters=516 \
    LeaningRate.base_lr=0.005 \
    YoloTrainFeed.batch_size=32 \
    LearningRate.schedulers='[!PiecewiseDecay {gamma: 0.1, milestones: [110000, 124000]}, !LinearWarmup {start_factor: 0., steps: 2000}]' \
    -d "../../dataset/voc"
```

通过`python compress.py --help`查看可配置参数。
通过`python ../../tools/configure.py ${option_name} help`查看如何通过命令行覆盖配置文件`yolov3_mobilenet_v1_voc.yml`中的参数。

### 保存断点（checkpoint）

如果在配置文件中设置了`checkpoint_path`, 则在压缩任务执行过程中会自动保存断点，当任务异常中断时，
重启任务会自动从`checkpoint_path`路径下按数字顺序加载最新的checkpoint文件。如果不想让重启的任务从断点恢复，
需要修改配置文件中的`checkpoint_path`，或者将`checkpoint_path`路径下文件清空。

>注意：配置文件中的信息不会保存在断点中，重启前对配置文件的修改将会生效。


## 评估

如果在配置文件中设置了`checkpoint_path`，则每个epoch会保存一个压缩后的用于评估的模型，
该模型会保存在`${checkpoint_path}/${epoch_id}/eval_model/`路径下，包含`__model__`和`__params__`两个文件。
其中，`__model__`用于保存模型结构信息，`__params__`用于保存参数（parameters）信息。

如果不需要保存评估模型，可以在定义Compressor对象时，将`save_eval_model`选项设置为False（默认为True）。

运行命令为：
```
python ../eval.py \
    --model_path ${checkpoint_path}/${epoch_id}/eval_model/ \
    --model_name __model__ \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc"
```

## 预测

如果在配置文件中设置了`checkpoint_path`，并且在定义Compressor对象时指定了`prune_infer_model`选项，则每个epoch都会
保存一个`inference model`。该模型是通过删除eval_program中多余的operators而得到的。

该模型会保存在`${checkpoint_path}/${epoch_id}/eval_model/`路径下，包含`__model__.infer`和`__params__`两个文件。
其中，`__model__.infer`用于保存模型结构信息，`__params__`用于保存参数（parameters）信息。

更多关于`prune_infer_model`选项的介绍，请参考：[Compressor介绍](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#121-%E5%A6%82%E4%BD%95%E6%94%B9%E5%86%99%E6%99%AE%E9%80%9A%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC)

### python预测

在脚本<a href="../infer.py">PaddleDetection/tools/infer.py</a>中展示了如何使用fluid python API加载使用预测模型进行预测。

运行命令为：
```
python ../infer.py \
    --model_path ${checkpoint_path}/${epoch_id}/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    --infer_dir ../../demo
```

### PaddleLite

该示例中产出的预测（inference）模型可以直接用PaddleLite进行加载使用。
关于PaddleLite如何使用，请参考：[PaddleLite使用文档](https://github.com/PaddlePaddle/Paddle-Lite/wiki#%E4%BD%BF%E7%94%A8)

## 示例结果

> 当前release的结果并非超参调优后的最好结果，仅做示例参考，后续我们会优化当前结果。

### MobileNetV1-YOLO-V3

| FLOPS |Box AP| model_size |Paddle Fluid inference time(ms)| Paddle Lite inference time(ms)|
|---|---|---|---|---|
|baseline|76.2 |93M |- |-|
|-50%|69.48 |51M |- |-|

## FAQ
