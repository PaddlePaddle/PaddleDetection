>运行该示例前请安装PaddleSlim和Paddle1.6或更高版本

# 模型蒸馏教程

## 概述

该示例使用PaddleSlim提供的[蒸馏策略](https://paddlepaddle.github.io/PaddleSlim/algo/algo/#3)对检测库中的模型进行蒸馏训练。
在阅读该示例前，建议您先了解以下内容：

- [检测库的常规训练方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)

已发布蒸馏模型见[压缩模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/slim/README.md)

## 安装PaddleSlim
可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim

## 蒸馏策略说明

关于蒸馏API如何使用您可以参考PaddleSlim蒸馏API文档

### MobileNetV1-YOLOv3在VOC数据集上的蒸馏

这里以ResNet34-YOLOv3蒸馏训练MobileNetV1-YOLOv3模型为例，首先，为了对`student model`和`teacher model`有个总体的认识，进一步确认蒸馏的对象，我们通过以下命令分别观察两个网络变量（Variables）的名称和形状：

```python
# 观察student model的Variables
student_vars = []
for v in fluid.default_main_program().list_vars():
    try:
        student_vars.append((v.name, v.shape))
    except:
        pass
print("="*50+"student_model_vars"+"="*50)
print(student_vars)
# 观察teacher model的Variables
teacher_vars = []
for v in teacher_program.list_vars():
    try:
        teacher_vars.append((v.name, v.shape))
    except:
        pass
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

### MobileNetV1-YOLOv3在COCO数据集上的蒸馏

这里以ResNet34-YOLOv3作为蒸馏训练的teacher网络, 对MobileNetV1-YOLOv3结构的student网络进行蒸馏。

COCO数据集作为目标检测任务的训练目标难度更大，意味着teacher网络会预测出更多的背景bbox，如果直接用teacher的预测输出作为student学习的`soft label`会有严重的类别不均衡问题。解决这个问题需要引入新的方法，详细背景请参考论文:[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1805.06361)

为了确定蒸馏的对象，我们首先需要找到student和teacher网络得到的`x,y,w,h,cls.objness`等变量在PaddlePaddle框架中的实际名称(var.name)。进而根据名称取出这些变量，用teacher得到的结果指导student训练。找到的所有变量如下：

```python
yolo_output_names = [
        'strided_slice_0.tmp_0', 'strided_slice_1.tmp_0',
        'strided_slice_2.tmp_0', 'strided_slice_3.tmp_0',
        'strided_slice_4.tmp_0', 'transpose_0.tmp_0', 'strided_slice_5.tmp_0',
        'strided_slice_6.tmp_0', 'strided_slice_7.tmp_0',
        'strided_slice_8.tmp_0', 'strided_slice_9.tmp_0', 'transpose_2.tmp_0',
        'strided_slice_10.tmp_0', 'strided_slice_11.tmp_0',
        'strided_slice_12.tmp_0', 'strided_slice_13.tmp_0',
        'strided_slice_14.tmp_0', 'transpose_4.tmp_0'
    ]
```

然后，就可以根据论文<<Object detection at 200 Frames Per Second>>的方法为YOLOv3中分类、回归、objness三个不同的head适配不同的蒸馏损失函数，并对分类和回归的损失函数用objness分值进行抑制，以解决前景背景类别不均衡问题。

## 训练

根据[PaddleDetection/tools/train.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/train.py)编写压缩脚本`distill.py`。
在该脚本中定义了teacher_model和student_model，用teacher_model的输出指导student_model的训练

### 执行示例

step1: 设置GPU卡

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

step2: 开始训练

```bash
# yolov3_mobilenet_v1在voc数据集上蒸馏
python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1_voc.yml \
    -t configs/yolov3_r34_voc.yml \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```

```bash
# yolov3_mobilenet_v1在COCO数据集上蒸馏
python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1.yml \
    -o use_fine_grained_loss=true \
    -t configs/yolov3_r34.yml \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
```

如果要调整训练卡数，需要调整配置文件`yolov3_mobilenet_v1_voc.yml`中的以下参数：

- **max_iters:** 训练过程迭代总步数。
- **YOLOv3Loss.batch_size:** 该参数表示单张GPU卡上的`batch_size`, 总`batch_size`是GPU卡数乘以这个值， `batch_size`的设定受限于显存大小。
- **LeaningRate.base_lr:** 根据多卡的总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。
- **LearningRate.schedulers.PiecewiseDecay.LinearWarmup.steps：** 请根据batch size的变化对其进行调整。

以下为4卡训练示例，通过命令行-o参数覆盖`yolov3_mobilenet_v1_voc.yml`中的参数, 修改GPU卡数后应尽量确保总batch_size(GPU卡数\*YoloTrainFeed.batch_size)不变, 以确保训练效果不因bs大小受影响:

```bash
# yolov3_mobilenet_v1在VOC数据集上蒸馏
CUDA_VISIBLE_DEVICES=0,1,2,3
python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1_voc.yml \
    -t configs/yolov3_r34_voc.yml \
    -o YOLOv3Loss.batch_size=16 \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```

```bash
# yolov3_mobilenet_v1在COCO数据集上蒸馏
CUDA_VISIBLE_DEVICES=0,1,2,3
python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1.yml \
    -t configs/yolov3_r34.yml \
    -o use_fine_grained_loss=true YOLOv3Loss.batch_size=16 \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
```



### 保存断点（checkpoint）

蒸馏任务执行过程中会自动保存断点。如果需要从断点继续训练请用`-r`参数指定checkpoint路径，示例如下：

```bash
# yolov3_mobilenet_v1在VOC数据集上恢复断点
python -u slim/distillation/distill.py \
-c configs/yolov3_mobilenet_v1_voc.yml \
-t configs/yolov3_r34_voc.yml \
-r output/yolov3_mobilenet_v1_voc/10000 \
--teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```

```bash
# yolov3_mobilenet_v1在COCO数据集上恢复断点
python -u slim/distillation/distill.py \
-c configs/yolov3_mobilenet_v1.yml \
-t configs/yolov3_r34.yml \
-o use_fine_grained_loss=true \
-r output/yolov3_mobilenet_v1/10000 \
--teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
```



## 评估

每隔`snap_shot_iter`步后会保存一个checkpoint模型可以用于评估，使用PaddleDetection目录下[tools/eval.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/eval.py)评估脚本，并指定`weights`为训练得到的模型路径

运行命令为：
```bash
# yolov3_mobilenet_v1在VOC数据集上评估
export CUDA_VISIBLE_DEVICES=0
python -u tools/eval.py -c configs/yolov3_mobilenet_v1_voc.yml \
       -o weights=output/yolov3_mobilenet_v1_voc/model_final \
```

```bash
# yolov3_mobilenet_v1在COCO数据集上评估
export CUDA_VISIBLE_DEVICES=0
python -u tools/eval.py -c configs/yolov3_mobilenet_v1.yml \
       -o weights=output/yolov3_mobilenet_v1/model_final \
```

## 预测

每隔`snap_shot_iter`步后保存的checkpoint模型也可以用于预测，使用PaddleDetection目录下[tools/infer.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/infer.py)评估脚本，并指定`weights`为训练得到的模型路径

### Python预测

运行命令为：
```
# 使用yolov3_mobilenet_v1_voc模型进行预测
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/yolov3_mobilenet_v1_voc.yml \
                    --infer_img=demo/000000570688.jpg \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5 \
                    -o weights=output/yolov3_mobilenet_v1_voc/model_final
```

```
# 使用yolov3_mobilenet_v1_coco模型进行预测
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/yolov3_mobilenet_v1.yml \
                    --infer_img=demo/000000570688.jpg \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5 \
                    -o weights=output/yolov3_mobilenet_v1/model_final
```

## 示例结果

### MobileNetV1-YOLO-V3-VOC

| FLOPS |输入尺寸|每张GPU图片个数|推理时间（fps）|Box AP|下载|
|:-:|:-:|:-:|:-:|:-:|:-:|
|baseline|608     |16|104.291|76.2|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|608 |16|106.914|79.0|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|
|baseline|416 |16|-|76.7|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|416 |16|-|78.2|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|
|baseline|320 |16|-|75.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|320 |16|-|75.5|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar)|

> 蒸馏后的结果用ResNet34-YOLO-V3做teacher，4GPU总batch_size64训练90000 iter得到

### MobileNetV1-YOLO-V3-COCO

| FLOPS |输入尺寸|每张GPU图片个数|推理时间（fps）|Box AP|下载|
|:-:|:-:|:-:|:-:|:-:|:-:|
|baseline|608     |16|78.302|29.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|608 |16|78.523|31.4|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|
|baseline|416 |16|-|29.3|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|416 |16|-|30.0|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|
|baseline|320 |16|-|27.0|[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar)|
|蒸馏后|320 |16|-|27.1|[下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar)|

> 蒸馏后的结果用ResNet34-YOLO-V3做teacher，4GPU总batch_size64训练600000 iter得到
