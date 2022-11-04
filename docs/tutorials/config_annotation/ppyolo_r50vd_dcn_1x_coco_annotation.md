# YOLO系列模型参数配置教程

标签： 模型参数配置

以`ppyolo_r50vd_dcn_1x_coco.yml`为例，这个模型由五个子配置文件组成：

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
    # 数据文件夹，os.path.join(dataset_dir, anno_path)
    dataset_dir: dataset/coco

TestDataset:
  !ImageFolder
    # 标注文件路径，相对 dataset_dir 路径
    anno_path: annotations/instances_val2017.json
```

- 优化器配置文件 `optimizer_1x.yml`

```yaml
# 总训练轮数
epoch: 405

# 学习率设置
LearningRate:
  # 默认为8卡训学习率
  base_lr: 0.01
  # 学习率调整策略
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    # 学习率变化位置(轮数)
    milestones:
    - 243
    - 324
  # Warmup
  - !LinearWarmup
    start_factor: 0.
    steps: 4000

# 优化器
OptimizerBuilder:
  # 优化器
  optimizer:
    momentum: 0.9
    type: Momentum
  # 正则化
  regularizer:
    factor: 0.0005
    type: L2
```

- 数据读取配置文件 `ppyolo_reader.yml`

```yaml
# 每张GPU reader进程个数
worker_num: 2
# 训练数据
TrainReader:
  inputs_def:
    num_max_boxes: 50
  # 训练数据transforms
  sample_transforms:
    - Decode: {}
    - Mixup: {alpha: 1.5, beta: 1.5}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
  # batch_transforms
  batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608], random_size: True, random_interp: True, keep_ratio: False}
    - NormalizeBox: {}
    - PadBox: {num_max_boxes: 50}
    - BboxXYXY2XYWH: {}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
    - Gt2YoloTarget: {anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]], anchors: [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], downsample_ratios: [32, 16, 8]}
  # 训练时batch_size
  batch_size: 24
  # 读取数据是否乱序
  shuffle: true
  # 是否丢弃最后不能完整组成batch的数据
  drop_last: true
  # mixup_epoch，大于最大epoch，表示训练过程一直使用mixup数据增广
  mixup_epoch: 25000
  # 是否通过共享内存进行数据读取加速，需要保证共享内存大小(如/dev/shm)满足大于1G
  use_shared_memory: true

# 评估数据
EvalReader:
  # 评估数据transforms
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  # 评估时batch_size
  batch_size: 8

# 测试数据
TestReader:
  inputs_def:
    image_shape: [3, 608, 608]
  # 测试数据transforms
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  # 测试时batch_size
  batch_size: 1
```

- 模型配置文件 `ppyolo_r50vd_dcn.yml`

```yaml
# 模型结构类型
architecture: YOLOv3
# 预训练模型地址
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
# norm_type
norm_type: sync_bn
# 是否使用ema
use_ema: true
# ema_decay
ema_decay: 0.9998

# YOLOv3
YOLOv3:
  # backbone
  backbone: ResNet
  # neck
  neck: PPYOLOFPN
  # yolo_head
  yolo_head: YOLOv3Head
  # post_process
  post_process: BBoxPostProcess


# backbone
ResNet:
  # depth
  depth: 50
  # variant
  variant: d
  # return_idx, 0 represent res2
  return_idx: [1, 2, 3]
  # dcn_v2_stages
  dcn_v2_stages: [3]
  # freeze_at
  freeze_at: -1
  # freeze_norm
  freeze_norm: false
  # norm_decay
  norm_decay: 0.

# PPYOLOFPN
PPYOLOFPN:
  # 是否coord_conv
  coord_conv: true
  # 是否drop_block
  drop_block: true
  # block_size
  block_size: 3
  # keep_prob
  keep_prob: 0.9
  # 是否spp
  spp: true

# YOLOv3Head
YOLOv3Head:
  # anchors
  anchors: [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]]
  # anchor_masks
  anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  # loss
  loss: YOLOv3Loss
  # 是否使用iou_aware
  iou_aware: true
  # iou_aware_factor
  iou_aware_factor: 0.4

# YOLOv3Loss
YOLOv3Loss:
  # ignore_thresh
  ignore_thresh: 0.7
  # downsample
  downsample: [32, 16, 8]
  # 是否label_smooth
  label_smooth: false
  # scale_x_y
  scale_x_y: 1.05
  # iou_loss
  iou_loss: IouLoss
  # iou_aware_loss
  iou_aware_loss: IouAwareLoss

# IouLoss
IouLoss:
  loss_weight: 2.5
  loss_square: true

# IouAwareLoss
IouAwareLoss:
  loss_weight: 1.0

# BBoxPostProcess
BBoxPostProcess:
  decode:
    name: YOLOBox
    conf_thresh: 0.01
    downsample_ratio: 32
    clip_bbox: true
    scale_x_y: 1.05
  # nms 配置
  nms:
    name: MatrixNMS
    keep_top_k: 100
    score_threshold: 0.01
    post_threshold: 0.01
    nms_top_k: -1
    background_label: -1

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
