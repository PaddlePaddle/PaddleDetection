>运行该示例前请安装Paddle1.6或更高版本

# 检测模型蒸馏示例

## 概述

该示例使用PaddleSlim提供的[蒸馏策略](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md#3-蒸馏)对检测库中的模型进行蒸馏训练。
在阅读该示例前，建议您先了解以下内容：

- [检测库的常规训练方法](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection)
- [PaddleSlim使用文档](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md)


## 配置文件说明

关于配置文件如何编写您可以参考：

- [PaddleSlim配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#122-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E7%9A%84%E4%BD%BF%E7%94%A8)
- [蒸馏策略配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#23-蒸馏)

这里以ResNet34-YoloV3蒸馏MobileNetV1-YoloV3模型为例，首先，为了对`student model`和`teacher model`有个总体的认识，从而进一步确认蒸馏的对象，我们通过以下命令分别观察两个网络变量（Variable）的名称和形状：

```python
# 观察student model的Variable
for v in fluid.default_main_program().list_vars():
    if "py_reader" not in v.name and "double_buffer" not in v.name and "generated_var" not in v.name:
        print(v.name, v.shape)
# 观察teacher model的Variable
for v in teacher_program.list_vars():
    print(v.name, v.shape)
```

经过对比可以发现，`student model`和`teacher model`的部分中间结果分别为：

```bash
# student model
conv2d_15.tmp_0
# teacher model
teacher_teacher_conv2d_1.tmp_0
```


所以，我们用`l2_distiller`对这两个特征图做蒸馏。在配置文件中进行如下配置：

```yaml
distillers:
    l2_distiller:
        class: 'L2Distiller'
        teacher_feature_map: 'teacher_teacher_conv2d_1.tmp_0'
        student_feature_map: 'conv2d_15.tmp_0'
        distillation_loss_weight: 1
strategies:
    distillation_strategy:
        class: 'DistillationStrategy'
        distillers: ['l2_distiller']
        start_epoch: 0
        end_epoch: 270
```

我们也可以根据上述操作为蒸馏策略选择其他loss，PaddleSlim支持的有`FSP_loss`, `L2_loss`和`softmax_with_cross_entropy_loss` 。

## 训练

根据[PaddleDetection/tools/train.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/PaddleDetection/tools/train.py)编写压缩脚本compress.py。
在该脚本中定义了Compressor对象，用于执行压缩任务。




您可以通过运行脚本`run.sh`运行该示例。


### 保存断点（checkpoint）

如果在配置文件中设置了`checkpoint_path`, 则在蒸馏任务执行过程中会自动保存断点，当任务异常中断时，
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

在脚本<a href="../infer.py">slim/infer.py</a>中展示了如何使用fluid python API加载使用预测模型进行预测。

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

>当前release的结果并非超参调优后的最好结果，仅做示例参考，后续我们会优化当前结果。

### MobileNetV1-YOLO-V3

| FLOPS |Box AP|
|---|---|
|baseline|76.2     |
|蒸馏后|- |


## FAQ
