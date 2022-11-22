# RCNN系列模型参数配置教程

标签： 模型参数配置

以`faster_rcnn_r50_fpn_1x_coco.yml`为例，这个模型由五个子配置文件组成：

- 数据配置文件 `coco_detection.yml`

```yaml
# 数据评估类型
metric: COCO
# 数据集的类别数
num_classes: 80

# TrainDataset
TrainDataset:
  !COCODataSet
    # 图像数据路径，相对 dataset_dir 路径，os.path.join(dataset_dir, image_dir)
    image_dir: train2017
    # 标注文件路径，相对 dataset_dir 路径，os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_train2017.json
    # 数据文件夹
    dataset_dir: dataset/coco
    # data_fields
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    # 图像数据路径，相对 dataset_dir 路径，os.path.join(dataset_dir, image_dir)
    image_dir: val2017
    # 标注文件路径，相对 dataset_dir 路径，os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_val2017.json
    # 数据文件夹
    dataset_dir: dataset/coco

TestDataset:
  !ImageFolder
    # 标注文件路径，相对 dataset_dir 路径，os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_val2017.json
```

- 优化器配置文件 `optimizer_1x.yml`

```yaml
# 总训练轮数
epoch: 12

# 学习率设置
LearningRate:
  # 默认为8卡训学习率
  base_lr: 0.01
  # 学习率调整策略
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    # 学习率变化位置(轮数)
    milestones: [8, 11]
  - !LinearWarmup
    start_factor: 0.1
    steps: 1000

# 优化器
OptimizerBuilder:
  # 优化器
  optimizer:
    momentum: 0.9
    type: Momentum
  # 正则化
  regularizer:
    factor: 0.0001
    type: L2
```

- 数据读取配置文件 `faster_fpn_reader.yml`

```yaml
# 每张GPU reader进程个数
worker_num: 2
# 训练数据
TrainReader:
  # 训练数据transforms
  sample_transforms:
  - Decode: {}
  - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], interp: 2, keep_ratio: True}
  - RandomFlip: {prob: 0.5}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # 由于模型存在FPN结构，输入图片需要padding为32的倍数
  - PadBatch: {pad_to_stride: 32}
  # 训练时batch_size
  batch_size: 1
  # 读取数据是否乱序
  shuffle: true
  # 是否丢弃最后不能完整组成batch的数据
  drop_last: true
  # 表示reader是否对gt进行组batch的操作，在rcnn系列算法中设置为false，得到的gt格式为list[Tensor]
  collate_batch: false

# 评估数据
EvalReader:
  # 评估数据transforms
  sample_transforms:
  - Decode: {}
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # 由于模型存在FPN结构，输入图片需要padding为32的倍数
  - PadBatch: {pad_to_stride: 32}
  # 评估时batch_size
  batch_size: 1
  # 读取数据是否乱序
  shuffle: false
  # 是否丢弃最后不能完整组成batch的数据
  drop_last: false

# 测试数据
TestReader:
  # 测试数据transforms
  sample_transforms:
  - Decode: {}
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # 由于模型存在FPN结构，输入图片需要padding为32的倍数
  - PadBatch: {pad_to_stride: 32}
  # 测试时batch_size
  batch_size: 1
  # 读取数据是否乱序
  shuffle: false
  # 是否丢弃最后不能完整组成batch的数据
  drop_last: false
```

- 模型配置文件 `faster_rcnn_r50_fpn.yml`

```yaml
# 模型结构类型
architecture: FasterRCNN
# 预训练模型地址
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams

# FasterRCNN
FasterRCNN:
  # backbone
  backbone: ResNet
  # neck
  neck: FPN
  # rpn_head
  rpn_head: RPNHead
  # bbox_head
  bbox_head: BBoxHead
  # post process
  bbox_post_process: BBoxPostProcess


# backbone
ResNet:
  # index 0 stands for res2
  depth: 50
  # norm_type，可设置参数：bn 或 sync_bn
  norm_type: bn
  # freeze_at index, 0 represent res2
  freeze_at: 0
  # return_idx
  return_idx: [0,1,2,3]
  # num_stages
  num_stages: 4

# FPN
FPN:
  # channel of FPN
  out_channel: 256

# RPNHead
RPNHead:
  # anchor generator
  anchor_generator:
    aspect_ratios: [0.5, 1.0, 2.0]
    anchor_sizes: [[32], [64], [128], [256], [512]]
    strides: [4, 8, 16, 32, 64]
  # rpn_target_assign
  rpn_target_assign:
    batch_size_per_im: 256
    fg_fraction: 0.5
    negative_overlap: 0.3
    positive_overlap: 0.7
    use_random: True
  # 训练时生成proposal的参数
  train_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 2000
    post_nms_top_n: 1000
    topk_after_collect: True
  # 评估时生成proposal的参数
  test_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 1000
    post_nms_top_n: 1000

# BBoxHead
BBoxHead:
  # TwoFCHead as BBoxHead
  head: TwoFCHead
  # roi align
  roi_extractor:
    resolution: 7
    sampling_ratio: 0
    aligned: True
  # bbox_assigner
  bbox_assigner: BBoxAssigner

# BBoxAssigner
BBoxAssigner:
  # batch_size_per_im
  batch_size_per_im: 512
  # 背景阈值
  bg_thresh: 0.5
  # 前景阈值
  fg_thresh: 0.5
  # 前景比例
  fg_fraction: 0.25
  # 是否随机采样
  use_random: True

# TwoFCHead
TwoFCHead:
  # TwoFCHead特征维度
  out_channel: 1024


# BBoxPostProcess
BBoxPostProcess:
  # 解码
  decode: RCNNBox
  # nms
  nms:
    # 使用MultiClassNMS
    name: MultiClassNMS
    keep_top_k: 100
    score_threshold: 0.05
    nms_threshold: 0.5

```

- 运行时置文件 `runtime.yml`

```yaml
# 是否使用gpu
use_gpu: true
# 日志打印间隔
log_iter: 20
# save_dir
save_dir: output
# 模型保存间隔时间
snapshot_epoch: 1
```
