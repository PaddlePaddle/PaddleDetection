# 蒸馏通道剪裁模型教程

该文档介绍如何使用[PaddleSlim](https://paddlepaddle.github.io/PaddleSlim)的蒸馏接口和卷积通道剪裁接口对检测库中的模型进行卷积层的通道剪裁并使用较高精度模型对其蒸馏。

在阅读该示例前，建议您先了解以下内容：

- [检测库的使用方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim通道剪裁API文档](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)
- [PaddleSlim蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)
- [检测库模型通道剪裁文档](../../prune/README.md)
- [检测库模型蒸馏文档](../../distillation/README.md)

请确保已正确[安装PaddleDetection](../../../docs/tutorials/INSTALL_cn.md)及其依赖。

已发布蒸馏通道剪裁模型见[压缩模型库](../../README.md)

蒸馏通道剪裁模型示例见[Ipython notebook示例](./distill_pruned_model_demo.ipynb)

## 1. 数据准备

请参考检测库[数据下载](../../../docs/tutorials/INSTALL_cn.md)文档准备数据。

## 2. 模型选择

通过`-c`选项指定待剪裁模型的配置文件的相对路径，更多可选配置文件请参考: [检测库配置文件](../../../configs)。

蒸馏通道剪裁模型中，我们使用原模型全量权重来初始化待剪裁模型，已发布模型的权重可在[模型库](../../../docs/MODEL_ZOO_cn.md)中获取。

通过`-o pretrain_weights`指定待剪裁模型的预训练权重，可以指定url或本地文件系统的路径。如下所示：

```
-o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar
```

或

```
-o pretrain_weights=output/yolov3_mobilenet_v1_voc/model_final
```

## 4. 启动蒸馏剪裁任务

使用`distill_pruned_model.py`启动蒸馏剪裁任务时，通过`--pruned_params`选项指定待剪裁的参数名称列表，参数名之间用空格分隔，通过`--pruned_ratios`选项指定各个参数被裁掉的比例。 获取待裁剪模型参数名称方法可参考[通道剪裁模教程](../../prune/README.md)。

通过`-t`参数指定teacher模型配置文件，`--teacher_pretrained`指定teacher模型权重，更多关于蒸馏模型设置可参考[模型蒸馏文档](../../distillation/README.md)。

蒸馏通道检测模型脚本目前只支持使用YOLOv3细粒度损失训练，即训练过程中须指定`-o use_fine_grained_loss=true`。

```
python distill_pruned_model.py \
-c ../../../configs/yolov3_mobilenet_v1_voc.yml \
-t ../../../configs/yolov3_r34_voc.yml \
--teacher_pretrained=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4" \
-o use_fine_grained_loss=true pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar
```

## 5. 评估模型

由于产出模型为通道剪裁模型，训练完成后，可通过通道剪裁中提供的评估脚本`../../prune/eval.py`评估模型精度，通过`--pruned_params`和`--pruned_ratios`指定剪裁的参数名称列表和各参数剪裁比例。

```
python ../../prune/eval.py \
-c ../../../configs/yolov3_mobilenet_v1_voc.yml \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4" \
-o weights=output/yolov3_mobilenet_v1_voc/model_final
```

## 6. 模型导出

如果想要将剪裁模型接入到C++预测库或者Serving服务，可通过`../../prune/export_model.py`导出该模型。

```
python ../../prune/export_model.py \
-c ../../../configs/yolov3_mobilenet_v1_voc.yml \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4" \
-o weights=output/yolov3_mobilenet_v1_voc/model_final
```
