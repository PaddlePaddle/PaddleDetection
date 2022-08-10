简体中文 | [English](./detection_en.md)

# 目标检测任务二次开发

在目标检测算法产业落地过程中，常常会出现需要额外训练以满足实际使用的要求，项目迭代过程中也会出先需要修改类别的情况。本文档详细介绍如何使用PaddleDetection进行目标检测算法二次开发，流程包括：数据准备、模型优化思路和修改类别开发流程。

## 数据准备

二次开发首先需要进行数据集的准备，针对场景特点采集合适的数据从而提升模型效果和泛化性能。然后使用Labeme，LabelImg等标注工具标注目标检测框，并将标注结果转化为COCO或VOC数据格式。详细文档可以参考[数据准备文档](../../tutorials/data/README.md)

## 模型优化

### 1. 使用自定义数据集训练

基于准备的数据在数据配置文件中修改对应路径，例如`configs/dataset/coco_detection.yml`:

```
metric: COCO
num_classes: 80

TrainDataset:
  !COCODataSet
    image_dir: train2017 # 训练集的图片所在文件相对于dataset_dir的路径
    anno_path: annotations/instances_train2017.json # 训练集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco # 数据集所在路径，相对于PaddleDetection路径
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val2017 # 验证集的图片所在文件相对于dataset_dir的路径
    anno_path: annotations/instances_val2017.json # 验证集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco # 数据集所在路径，相对于PaddleDetection路径

TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val2017.json # also support txt (like VOC's label_list.txt) # 标注文件所在文件 相对于dataset_dir的路径
    dataset_dir: dataset/coco # if set, anno_path will be 'dataset_dir/anno_path' # 数据集所在路径，相对于PaddleDetection路径
```

配置修改完成后，即可以启动训练评估，命令如下

```
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml --eval
```

更详细的命令参考[30分钟快速上手PaddleDetection](../../tutorials/GETTING_STARTED_cn.md)


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
      max_epochs: 144 # 依据epoch数进行修改
    - !LinearWarmup
      start_factor: 0.
      epochs: 5
```

## 修改类别

当实际使用场景类别发生变化时，需要修改数据配置文件，例如`configs/datasets/coco_detection.yml`中:

```
metric: COCO
num_classes: 10 # 原始类别80
```

配置修改完成后，同样可以加载COCO预训练权重，PaddleDetection支持自动加载shape匹配的权重，对于shape不匹配的权重会自动忽略，因此无需其他修改。
