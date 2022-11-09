# YOLO series model parameter configuration tutorial

Tag: Model parameter configuration

Take `ppyolo_r50vd_dcn_1x_coco.yml` as an example, The model consists of five sub-profiles:

- Data profile `coco_detection.yml`

```yaml
# Data evaluation type
metric: COCO
# The number of categories in the dataset
num_classes: 80

# TrainDataset
TrainDataset:
  !COCODataSet
    # Image data path, Relative path of dataset_dir, os.path.join(dataset_dir, image_dir)
    image_dir: train2017
    # Annotation file path, Relative path of dataset_dir, os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_train2017.json
    # data file
    dataset_dir: dataset/coco
    # data_fields
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    # Image data path, Relative path of dataset_dir, os.path.join(dataset_dir, image_dir)
    image_dir: val2017
    # Annotation file path, Relative path of dataset_dir, os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_val2017.json
    # data file os.path.join(dataset_dir, anno_path)
    dataset_dir: dataset/coco

TestDataset:
  !ImageFolder
    # Annotation file path, Relative path of dataset_dir, os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_val2017.json
```

- Optimizer configuration file `optimizer_1x.yml`

```yaml
# Total training epoches
epoch: 405

# learning rate setting
LearningRate:
  # Default is 8 Gpus training learning rate
  base_lr: 0.01
  # Learning rate adjustment strategy
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    # Position of change in learning rate (number of epoches)
    milestones:
    - 243
    - 324
  # Warmup
  - !LinearWarmup
    start_factor: 0.
    steps: 4000

# Optimizer
OptimizerBuilder:
  # Optimizer
  optimizer:
    momentum: 0.9
    type: Momentum
  # Regularization
  regularizer:
    factor: 0.0005
    type: L2
```

- Data reads configuration files `ppyolo_reader.yml`

```yaml
# Number of PROCESSES per GPU Reader
worker_num: 2
# training data
TrainReader:
  inputs_def:
    num_max_boxes: 50
  # Training data transforms
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
  # Batch size during training
  batch_size: 24
  # Read data is out of order
  shuffle: true
  # Whether to discard data that does not complete the batch
  drop_last: true
  # mixup_epoch，Greater than maximum epoch, Indicates that the training process has been augmented with mixup data
  mixup_epoch: 25000
  # Whether to use the shared memory to accelerate data reading, ensure that the shared memory size (such as /dev/shm) is greater than 1 GB
  use_shared_memory: true

# Evaluate data
EvalReader:
  # Evaluating data transforms
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  # Batch_size during evaluation
  batch_size: 8

# test data
TestReader:
  inputs_def:
    image_shape: [3, 608, 608]
  # test data transforms
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  # batch_size during training
  batch_size: 1
```

- Model profile `ppyolo_r50vd_dcn.yml`

```yaml
# Model structure type
architecture: YOLOv3
# Pretrain model address
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
# norm_type
norm_type: sync_bn
# Whether to use EMA
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
  # whether coord_conv or not
  coord_conv: true
  # whether drop_block or not
  drop_block: true
  # block_size
  block_size: 3
  # keep_prob
  keep_prob: 0.9
  # whether spp or not
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
  # whether to use iou_aware
  iou_aware: true
  # iou_aware_factor
  iou_aware_factor: 0.4

# YOLOv3Loss
YOLOv3Loss:
  # ignore_thresh
  ignore_thresh: 0.7
  # downsample
  downsample: [32, 16, 8]
  # whether label_smooth or not
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
  # nms setting
  nms:
    name: MatrixNMS
    keep_top_k: 100
    score_threshold: 0.01
    post_threshold: 0.01
    nms_top_k: -1
    background_label: -1

```

- Runtime file `runtime.yml`

```yaml
# Whether to use gpu
use_gpu: true
# Log Printing interval
log_iter: 20
# save_dir
save_dir: output
# Model save interval
snapshot_epoch: 1
```
