简体中文 | [English](./pphuman_mot_en.md)

# 多目标跟踪任务二次开发

在产业落地过程中应用多目标跟踪算法，不可避免地会出现希望自定义类型的多目标跟踪的需求，或是对已有多目标跟踪模型的优化，以提升在特定场景下模型的效果。我们在本文档通过案例来介绍如何根据期望识别的行为来进行多目标跟踪方案的选择，以及使用PaddleDetection进行多目标跟踪算法二次开发工作，包括：数据准备、模型优化思路和跟踪类别修改的开发流程。

## 数据准备

多目标跟踪模型方案采用[ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)，其中使用PP-YOLOE替换原文的YOLOX作为检测器，使用BYTETracker作为跟踪器，详细文档参考[ByteTrack](../../../configs/mot/bytetrack)。原文的ByteTrack只支持行人单类别，PaddleDetection中也支持多类别同时进行跟踪。训练ByteTrack也就是训练检测器的过程，只需要准备好检测标注即可，不需要ReID标注信息，即当成纯检测来做即可。数据集最好是连续视频中抽取出来的而不是无关联的图片集合。
二次开发首先需要进行数据集的准备，针对场景特点采集合适的数据从而提升模型效果和泛化性能。然后使用Labeme，LabelImg等标注工具标注目标检测框，并将标注结果转化为COCO或VOC数据格式。详细文档可以参考[数据准备文档](../../tutorials/data/README.md)

## 模型优化

### 1. 使用自定义数据集训练

ByteTrack跟踪方案采用的数据集只需要有检测标注即可。参照[MOT数据集准备](../../../configs/mot)和[MOT数据集教程](docs/tutorials/data/PrepareMOTDataSet.md)。

```
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --eval --amp

# 多卡训练
python -m paddle.distributed.launch --log_dir=log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --eval --amp
```

更详细的命令参考[30分钟快速上手PaddleDetection](../../tutorials/GETTING_STARTED_cn.md)和[ByteTrack](../../../configs/mot/bytetrack/detector)


### 2. 加载COCO模型作为预训练

目前PaddleDetection提供的配置文件加载的预训练模型均为ImageNet数据集的权重，加载到检测算法的骨干网络中，实际使用时，建议加载COCO数据集训练好的权重，通常能够对模型精度有较大提升，使用方法如下：

#### 1) 设置预训练权重路径

COCO数据集训练好的模型权重均在各算法配置文件夹下，例如`configs/ppyoloe`下提供了PP-YOLOE-l COCO数据集权重：[链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) 。配置文件中设置`pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams`

#### 2) 修改超参数

加载COCO预训练权重后，需要修改学习率超参数，例如`configs/ppyoloe/_base_/optimizer_300e.yml`中:

```
epoch: 120 # 原始配置为300epoch，加载COCO权重后可以适当减少迭代轮数

LearningRate:
  base_lr: 0.005 # 原始配置为0.025，加载COCO权重后需要降低学习率
  schedulers:
    - !CosineDecay
      max_epochs: 144 # 依据epoch数进行修改，一般为epoch数的1.2倍
    - !LinearWarmup
      start_factor: 0.
      epochs: 5
```

## 跟踪类别修改

当实际使用场景类别发生变化时，需要修改数据配置文件，例如`configs/datasets/coco_detection.yml`中:

```
metric: COCO
num_classes: 10 # 原始类别1
```

配置修改完成后，同样可以加载COCO预训练权重，PaddleDetection支持自动加载shape匹配的权重，对于shape不匹配的权重会自动忽略，因此无需其他修改。
