>运行该示例前请安装Paddle1.6或更高版本

# 检测模型量化压缩示例

## 概述

该示例使用PaddleSlim提供的[量化压缩策略](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md#1-quantization-aware-training%E9%87%8F%E5%8C%96%E4%BB%8B%E7%BB%8D)对分类模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [检测模型的常规训练方法](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection)
- [PaddleSlim使用文档](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md)


## 配置文件说明

关于配置文件如何编写您可以参考：

- [PaddleSlim配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#122-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E7%9A%84%E4%BD%BF%E7%94%A8)
- [量化策略配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#21-%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83)

其中save_out_nodes需要得到检测结果的Variable的名称，下面介绍如何确定save_out_nodes的参数
以MobileNet V1为例，可在compress.py中构建好网络之后，直接打印Variable得到Variable的名称信息。
代码示例：
```
    eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                         extra_keys)
    # print(eval_values)
```
根据运行结果可看到Variable的名字为：`multiclass_nms_0.tmp_0`。
## 训练

根据 [PaddleCV/PaddleDetection/tools/train.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/PaddleDetection/tools/train.py) 编写压缩脚本compress.py。
在该脚本中定义了Compressor对象，用于执行压缩任务。

通过`python compress.py --help`查看可配置参数，简述如下：

- config: 检测库的配置，其中配置了训练超参数、数据集信息等。
- slim_file: PaddleSlim的配置文件，参见[配置文件说明](#配置文件说明)。

您可以通过运行以下命令运行该示例，请确保已正确下载[pretrained model](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E5%B7%B2%E5%8F%91%E5%B8%83%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E6%80%A7%E8%83%BD)。

step1: 开启显存优化策略
```
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
```
step2: 设置gpu卡,目前的超参设置适合2卡训练
```
export CUDA_VISIBLE_DEVICES=0,1
```
step3: 开始训练
```
python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c yolov3_mobilenet_v1_voc.yml 
```

### 训练时的模型结构
这部分介绍来源于[量化low-level API介绍](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api#1-%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83low-level-apis%E4%BB%8B%E7%BB%8D)。

PaddlePaddle框架中有四个和量化相关的IrPass, 分别是QuantizationTransformPass、QuantizationFreezePass、ConvertToInt8Pass以及TransformForMobilePass。在训练时，对网络应用了QuantizationTransformPass，作用是在网络中的conv2d、depthwise_conv2d、mul等算子的各个输入前插入连续的量化op和反量化op，并改变相应反向算子的某些输入。示例图如下：

<p align="center">
<img src="./images/TransformPass.png" height=400 width=520 hspace='10'/> <br />
<strong>图1：应用QuantizationTransformPass后的结果</strong>
</p>

### 保存断点（checkpoint）

如果在配置文件中设置了`checkpoint_path`, 则在压缩任务执行过程中会自动保存断点，当任务异常中断时，
重启任务会自动从`checkpoint_path`路径下按数字顺序加载最新的checkpoint文件。如果不想让重启的任务从断点恢复，
需要修改配置文件中的`checkpoint_path`，或者将`checkpoint_path`路径下文件清空。

>注意：配置文件中的信息不会保存在断点中，重启前对配置文件的修改将会生效。


### 保存评估和预测模型

如果在配置文件的量化策略中设置了`float_model_save_path`, `int8_model_save_path`, `mobile_model_save_path`, 在训练结束后，会保存模型量化压缩之后用于预测的模型。接下来介绍这三种预测模型的区别。

#### FP32模型
在介绍量化训练时的模型结构时介绍了PaddlePaddle框架中有四个和量化相关的IrPass, 分别是QuantizationTransformPass、QuantizationFreezePass、ConvertToInt8Pass以及TransformForMobilePass。FP32模型是在应用QuantizationFreezePass并删除eval_program中多余的operators之后，保存的模型。

QuantizationFreezePass主要用于改变IrGraph中量化op和反量化op的顺序，即将类似图1中的量化op和反量化op顺序改变为图2中的布局。除此之外，QuantizationFreezePass还会将`conv2d`、`depthwise_conv2d`、`mul`等算子的权重离线量化为int8_t范围内的值(但数据类型仍为float32)，以减少预测过程中对权重的量化操作，示例如图2：

<p align="center">
<img src="./images/FreezePass.png" height=400 width=420 hspace='10'/> <br />
<strong>图2：应用QuantizationFreezePass后的结果</strong>
</p>

#### 8-bit模型
在对训练网络进行QuantizationFreezePass之后，执行ConvertToInt8Pass，
其主要目的是将执行完QuantizationFreezePass后输出的权重类型由`FP32`更改为`INT8`。换言之，用户可以选择将量化后的权重保存为float32类型（不执行ConvertToInt8Pass）或者int8_t类型（执行ConvertToInt8Pass），示例如图3：

<p align="center">
<img src="./images/ConvertToInt8Pass.png" height=400 width=400 hspace='10'/> <br />
<strong>图3：应用ConvertToInt8Pass后的结果</strong>
</p>

####  mobile模型
经TransformForMobilePass转换后，用户可得到兼容[paddle-lite](https://github.com/PaddlePaddle/Paddle-Lite)移动端预测库的量化模型。paddle-mobile中的量化op和反量化op的名称分别为`quantize`和`dequantize`。`quantize`算子和PaddlePaddle框架中的`fake_quantize_abs_max`算子簇的功能类似，`dequantize` 算子和PaddlePaddle框架中的`fake_dequantize_max_abs`算子簇的功能相同。若选择paddle-mobile执行量化训练输出的模型，则需要将`fake_quantize_abs_max`等算子改为`quantize`算子以及将`fake_dequantize_max_abs`等算子改为`dequantize`算子，示例如图4：

<p align="center">
<img src="./images/TransformForMobilePass.png" height=400 width=400 hspace='10'/> <br />
<strong>图4：应用TransformForMobilePass后的结果</strong>
</p>
> 综上，可得在量化过程中有以下几种模型结构：
1. 原始模型
2. 经QuantizationTransformPass之后得到的适用于训练的量化模型结构，在${checkpoint_path}下保存的`eval_model`是这种结构，在训练过程中每个epoch结束时也使用这个网络结构进行评估，虽然这个模型结构不是最终想要的模型结构，但是每个epoch的评估结果可用来挑选模型。
3. 经QuantizationFreezePass之后得到的FP32模型结构，具体结构已在上面进行介绍。本文档中列出的数据集的评估结果是对FP32模型结构进行评估得到的结果。这种模型结构在训练过程中只会保存一次，也就是在量化配置文件中设置的`end_epoch`结束时进行保存，如果想将其他epoch的训练结果转化成FP32模型，可使用脚本 <a href='./freeze.py'>PaddleSlim/classification/quantization/freeze.py</a>进行转化，具体使用方法在[评估](#评估)中介绍。
4. 经ConvertToInt8Pass之后得到的8-bit模型结构，具体结构已在上面进行介绍。这种模型结构在训练过程中只会保存一次，也就是在量化配置文件中设置的`end_epoch`结束时进行保存，如果想将其他epoch的训练结果转化成8-bit模型，可使用脚本 <a href='./freeze.py'>slim/quantization/freeze.py</a>进行转化，具体使用方法在[评估](#评估)中介绍。
5. 经TransformForMobilePass之后得到的mobile模型结构，具体结构已在上面进行介绍。这种模型结构在训练过程中只会保存一次，也就是在量化配置文件中设置的`end_epoch`结束时进行保存，如果想将其他epoch的训练结果转化成mobile模型，可使用脚本 <a href='./freeze.py'>slim/quantization/freeze.py</a>进行转化，具体使用方法在[评估](#评估)中介绍。

## 评估

### 每个epoch保存的评估模型
因为量化的最终模型只有在end_epoch时保存一次，不能保证保存的模型是最好的，因此
如果在配置文件中设置了`checkpoint_path`，则每个epoch会保存一个量化后的用于评估的模型，
该模型会保存在`${checkpoint_path}/${epoch_id}/eval_model/`路径下，包含`__model__`和`__params__`两个文件。
其中，`__model__`用于保存模型结构信息，`__params__`用于保存参数（parameters）信息。模型结构和训练时一样。

如果不需要保存评估模型，可以在定义Compressor对象时，将`save_eval_model`选项设置为False（默认为True）。

脚本<a href="../eval.py">slim/eval.py</a>中为使用该模型在评估数据集上做评估的示例。
运行命令为：
```
python ../eval.py \
    --model_path ${checkpoint_path}/${epoch_id}/eval_model/ \
    --model_name __model__ \
    --params_name __params__ \
    -c yolov3_mobilenet_v1_voc.yml
```

在评估之后，选取效果最好的epoch的模型，可使用脚本 <a href='./freeze.py'>slim/quantization/freeze.py</a>将该模型转化为以上介绍的三种模型：FP32模型，int8模型，mobile模型，需要配置的参数为：

- model_path, 加载的模型路径，`为${checkpoint_path}/${epoch_id}/eval_model/`
- weight_quant_type 模型参数的量化方式，和配置文件中的类型保持一致
- save_path `FP32`, `8-bit`, `mobile`模型的保存路径，分别为 `${save_path}/float/`, `${save_path}/int8/`, `${save_path}/mobile/`

运行命令示例：
```
python freeze.py \
    --model_path ${checkpoint_path}/${epoch_id}/eval_model/ \
    --weight_quant_type ${weight_quant_type} \
    --save_path ${any path you want}
```

### 最终评估模型
最终使用的评估模型是FP32模型，使用脚本<a href="../eval.py">slim/eval.py</a>中为使用该模型在评估数据集上做评估的示例。
运行命令为：
```
python ../eval.py \
    --model_path ${float_model_path} 
    --model_name model \
    --params_name weights \
    -c yolov3_mobilenet_v1_voc.yml
```

## 预测

### python预测
FP32模型可直接使用原生PaddlePaddle Fluid预测方法进行预测。

在脚本<a href="../infer.py">slim/infer.py</a>中展示了如何使用fluid python API加载使用预测模型进行预测。

运行命令示例:
```
python ../infer.py \
    --model_path ${save_path}/float \
    --model_name model \
    --params_name weights \
    -c yolov3_mobilenet_v1_voc.yml \
    --infer_dir ../../demo
```


### PaddleLite预测
FP32模型可使用PaddleLite进行加载预测，可参见教程[Paddle-Lite如何加载运行量化模型](https://github.com/PaddlePaddle/Paddle-Lite/wiki/model_quantization)


## 示例结果

### MobileNetV1

| weight量化方式 | activation量化方式| Box ap |Paddle Fluid inference time(ms)| Paddle Lite inference time(ms)|
|---|---|---|---|---|
|baseline|- |76.2%|- |-|
|abs_max|abs_max|- |- |-|
|abs_max|moving_average_abs_max|- |- |-|
|channel_wise_abs_max|abs_max|- |- |-|

>训练超参：


## FAQ


