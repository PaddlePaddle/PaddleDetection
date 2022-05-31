# 卷积层通道剪裁教程

请确保已正确[安装PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/docs/tutorials/INSTALL_cn.md)及其依赖。

该文档介绍如何使用[PaddleSlim](https://paddlepaddle.github.io/PaddleSlim)的卷积通道剪裁接口对检测库中的模型的卷积层的通道数进行剪裁。

在检测库中，可以直接调用`PaddleDetection/slim/prune/prune.py`脚本实现剪裁，在该脚本中调用了PaddleSlim的[paddleslim.prune.Pruner](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#Pruner)接口。

该教程中所示操作，如无特殊说明，均在`PaddleDetection/`路径下执行。

已发布裁剪模型见[压缩模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/slim/README.md)

## 1. 数据准备

请参考检测库[数据下载](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/docs/tutorials/INSTALL_cn.md)文档准备数据。

## 2. 模型选择

通过`-c`选项指定待裁剪模型的配置文件的相对路径，更多可选配置文件请参考: [检测库配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs)

对于剪裁任务，原模型的权重不一定对剪裁后的模型训练的重训练有贡献，所以加载原模型的权重不是必需的步骤。

通过`-o pretrain_weights`指定模型的预训练权重，可以指定url或本地文件系统的路径。如下所示：

```
-o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar
```

或

```
-o weights=output/yolov3_mobilenet_v1_voc/model_final
```

官方已发布的模型请参考: [模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/docs/MODEL_ZOO_cn.md)

## 3. 确定待分析参数

我们通过剪裁卷积层参数达到缩减卷积层通道数的目的，在剪裁之前，我们需要确定待裁卷积层的参数的名称。
通过以下命令查看当前模型的所有参数：

```
python slim/prune/prune.py \
-c ./configs/yolov3_mobilenet_v1_voc.yml \
--print_params
```

通过观察参数名称和参数的形状，筛选出所有卷积层参数，并确定要裁剪的卷积层参数。

## 4. 分析待剪裁参数敏感度

可通过敏感度分析脚本分析待剪裁参数敏感度得到合适的剪裁率，敏感度分析工具见[敏感度分析](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/slim/sensitive/README.md)。

## 5. 启动剪裁任务

使用`prune.py`启动裁剪任务时，通过`--pruned_params`选项指定待裁剪的参数名称列表，参数名之间用空格分隔，通过`--pruned_ratios`选项指定各个参数被裁掉的比例。

```
python slim/prune/prune.py \
-c ./configs/yolov3_mobilenet_v1_voc.yml \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4"
```

## 6. 评估剪裁模型

训练剪裁任务完成后，可通过`eval.py`评估剪裁模型精度，通过`--pruned_params`和`--pruned_ratios`指定裁剪的参数名称列表和各参数裁剪比例。

```
python slim/prune/eval.py \
-c ./configs/yolov3_mobilenet_v1_voc.yml \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4" \
-o weights=output/yolov3_mobilenet_v1_voc/model_final
```

## 7. 模型导出

如果想要将剪裁模型接入到C++预测库或者Serving服务，可通过`export_model.py`导出该模型。

```
python slim/prune/export_model.py \
-c ./configs/yolov3_mobilenet_v1_voc.yml \
--pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
--pruned_ratios="0.2,0.3,0.4" \
-o weights=output/yolov3_mobilenet_v1_voc/model_final
```

## 8. 扩展模型

如果需要对自己的模型进行修改，可以参考`prune.py`中对`paddleslim.prune.Pruner`接口的调用方式，基于自己的模型训练脚本进行修改。
本节我们介绍的剪裁示例，需要用户根据先验知识指定每层的剪裁率，除此之外，PaddleSlim还提供了敏感度分析等功能，协助用户选择合适的剪裁率。更多详情请参考：[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)

## 9. 更多示例与注意事项

## 9.1 faster_rcnn与mask_rcnn

**当前PaddleSlim的剪裁功能不支持剪裁循环体或条件判断语句块内的卷积层，请避免剪裁循环和判断语句块前的一个卷积和语句块内部的卷积。**

对于[faster_rcnn_r50](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/faster_rcnn_r50_1x.yml)或[mask_rcnn_r50](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/mask_rcnn_r50_1x.yml)网络，请剪裁卷积`res4f_branch2c`之前的卷积。

对[faster_rcnn_r50](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/faster_rcnn_r50_1x.yml)剪裁示例如下：

```
# demo for faster_rcnn_r50
python slim/prune/prune.py -c ./configs/faster_rcnn_r50_1x.yml --pruned_params "res4f_branch2b_weights,res4f_branch2a_weights" --pruned_ratios="0.3,0.4" --eval
```

对[mask_rcnn_r50](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/mask_rcnn_r50_1x.yml)剪裁示例如下：

```
# demo for mask_rcnn_r50
python slim/prune/prune.py -c ./configs/mask_rcnn_r50_1x.yml --pruned_params "res4f_branch2b_weights,res4f_branch2a_weights" --pruned_ratios="0.2,0.3" --eval

```
