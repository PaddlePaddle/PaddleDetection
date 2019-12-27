>运行该示例前请安装PaddleSlim和Paddle1.6或更高版本

# 检测模型蒸馏示例

## 概述

该示例使用PaddleSlim提供的[蒸馏策略](https://paddlepaddle.github.io/PaddleSlim/algo/algo/#3)对检测库中的模型进行蒸馏训练。
在阅读该示例前，建议您先了解以下内容：

- [检测库的常规训练方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)


## 蒸馏策略说明

关于蒸馏API如何使用您可以参考PaddleSlim蒸馏API文档

这里以ResNet34-YoloV3蒸馏训练MobileNetV1-YoloV3模型为例，首先，为了对`student model`和`teacher model`有个总体的认识，进一步确认蒸馏的对象，我们通过以下命令分别观察两个网络变量（Variables）的名称和形状：

```python
# 观察student model的Variables
student_vars = []
for v in fluid.default_main_program().list_vars():
    if "py_reader" not in v.name and "double_buffer" not in v.name and "generated_var" not in v.name:
        student_vars.append((v.name, v.shape))
print("="*50+"student_model_vars"+"="*50)
print(student_vars)
# 观察teacher model的Variables
teacher_vars = []
for v in teacher_program.list_vars():
    teacher_vars.append((v.name, v.shape))
print("="*50+"teacher_model_vars"+"="*50)
print(teacher_vars)
```

经过对比可以发现，`student model`和`teacher model`输入到3个`yolov3_loss`的特征图分别为：

```bash
# student model
conv2d_20.tmp_1, conv2d_28.tmp_1, conv2d_36.tmp_1
# teacher model
conv2d_6.tmp_1, conv2d_14.tmp_1, conv2d_22.tmp_1
```


它们形状两两相同，且分别处于两个网络的输出部分。所以，我们用`l2_loss`对这几个特征图两两对应添加蒸馏loss。需要注意的是，teacher的Variable在merge过程中被自动添加了一个`name_prefix`，所以这里也需要加上这个前缀`"teacher_"`，merge过程请参考[蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/#merge)

```python
dist_loss_1 = l2_loss('teacher_conv2d_6.tmp_1', 'conv2d_20.tmp_1')
dist_loss_2 = l2_loss('teacher_conv2d_14.tmp_1', 'conv2d_28.tmp_1')
dist_loss_3 = l2_loss('teacher_conv2d_22.tmp_1', 'conv2d_36.tmp_1')
```

我们也可以根据上述操作为蒸馏策略选择其他loss，PaddleSlim支持的有`FSP_loss`, `L2_loss`, `softmax_with_cross_entropy_loss` 以及自定义的任何loss。

## 训练

根据[PaddleDetection/tools/train.py](../../tools/train.py)编写压缩脚本distill.py。
在该脚本中定义了teacher_model和student_model，用teacher_model的输出指导student_model的训练

### 执行示例

step1: 设置GPU卡

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

step2: 开始训练

```bash
python distill.py \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -t ../../configs/yolov3_r34_voc.yml \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```

如果要调整训练卡数，需要调整配置文件`yolov3_mobilenet_v1_voc.yml`中的以下参数：

- **max_iters:** 一个`epoch`中batch的数量，需要设置为`total_num / batch_size`, 其中`total_num`为训练样本总数量，`batch_size`为多卡上总的batch size.
- **YOLOv3Loss.batch_size:** 该参数表示单张GPU卡上的`batch_size`, 总`batch_size`是GPU卡数乘以这个值， `batch_size`的设定受限于显存大小。
- **LeaningRate.base_lr:** 根据多卡的总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。
- **LearningRate.schedulers.PiecewiseDecay.LinearWarmup.steps：** 请根据batch size的变化对其进行调整。

以下为4卡训练示例，通过命令行覆盖`yolov3_mobilenet_v1_voc.yml`中的参数：

```shell
python distill.py \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -t ../../configs/yolov3_r34_voc.yml \
    -o YoloTrainFeed.batch_size=16 \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```




### 保存断点（checkpoint）

蒸馏任务执行过程中会自动保存断点。如果需要从断点继续训练请用`-r`参数指定checkpoint路径，示例如下：

```bash
python -u distill.py \
-c ../../configs/yolov3_mobilenet_v1_voc.yml \
-t ../../configs/yolov3_r34_voc.yml \
-r output/yolov3_mobilenet_v1_voc/10000 \
--teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```




## 评估

每隔`snap_shot_iter`步后会保存一个checkpoint模型可以用于评估，使用PaddleDetection目录下[tools/eval.py](../../tools/eval.py)评估脚本，并指定`weights`为训练得到的模型路径

运行命令为：
```bash
export CUDA_VISIBLE_DEVICES=0
python -u ../../tools/eval.py -c ../../configs/yolov3_mobilenet_v1_voc.yml \
       -o weights=output/yolov3_mobilenet_v1_voc/model_final \
```

## 预测

每隔`snap_shot_iter`步后保存的checkpoint模型也可以用于预测，使用PaddleDetection目录下[tools/infer.py](../../tools/infer.py)评估脚本，并指定`weights`为训练得到的模型路径

### Python预测

运行命令为：
```
export CUDA_VISIBLE_DEVICES=0
python -u ../../tools/infer.py -c ../../configs/yolov3_mobilenet_v1_voc.yml \
                    --infer_img=demo/000000570688.jpg \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5 \
                    -o weights=output/yolov3_mobilenet_v1_voc/model_final
```

## 示例结果

### MobileNetV1-YOLO-V3-VOC

| FLOPS |Box AP|
|---|---|
|baseline|76.2     |
|蒸馏后|79.0（+2.8） |

> 蒸馏后的结果用ResNet34-YOLO-V3做teacher，4GPU总batch_size64训练90000 iter得到

## FAQ
