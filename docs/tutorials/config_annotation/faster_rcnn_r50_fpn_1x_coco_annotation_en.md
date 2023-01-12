# RCNN series model parameter configuration tutorial

Tag: Model parameter configuration

Take `faster_rcnn_r50_fpn_1x_coco.yml` as an example. The model consists of five sub-profiles:

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
    # data file file os.path.join(dataset_dir, anno_path)
    dataset_dir: dataset/coco

TestDataset:
  !ImageFolder
    # Annotation file path, Relative path of dataset_dir, os.path.join(dataset_dir, anno_path)
    anno_path: annotations/instances_val2017.json
```

- Optimizer configuration file `optimizer_1x.yml`

```yaml
# Total training epoches
epoch: 12

# learning rate setting
LearningRate:
  # Default is 8 Gpus training learning rate
  base_lr: 0.01
  # Learning rate adjustment strategy
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    # Position of change in learning rate (number of epoches)
    milestones: [8, 11]
  - !LinearWarmup
    start_factor: 0.1
    steps: 1000

# Optimizer
OptimizerBuilder:
  # Optimizer
  optimizer:
    momentum: 0.9
    type: Momentum
  # Regularization
  regularizer:
    factor: 0.0001
    type: L2
```

- Data reads configuration files `faster_fpn_reader.yml`

```yaml
# Number of PROCESSES per GPU Reader
worker_num: 2
# training data
TrainReader:
  # Training data transforms
  sample_transforms:
  - Decode: {}
  - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], interp: 2, keep_ratio: True}
  - RandomFlip: {prob: 0.5}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # Since the model has FPN structure, the input image needs a multiple of 32 padding
  - PadBatch: {pad_to_stride: 32}
  # Batch_size during training
  batch_size: 1
  # Read data is out of order
  shuffle: true
  # Whether to discard data that does not complete the batch
  drop_last: true
  # Set it to false. Then you have a sequence of values for GT: List [Tensor]
  collate_batch: false

# Evaluate data
EvalReader:
  # Evaluate data transforms
  sample_transforms:
  - Decode: {}
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # Since the model has FPN structure, the input image needs a multiple of 32 padding
  - PadBatch: {pad_to_stride: 32}
  # batch_size of evaluation
  batch_size: 1
  # Read data is out of order
  shuffle: false
  # Whether to discard data that does not complete the batch
  drop_last: false

# test data
TestReader:
  # test data transforms
  sample_transforms:
  - Decode: {}
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
  batch_transforms:
  # Since the model has FPN structure, the input image needs a multiple of 32 padding
  - PadBatch: {pad_to_stride: 32}
  # batch_size of test
  batch_size: 1
  # Read data is out of order
  shuffle: false
  # Whether to discard data that does not complete the batch
  drop_last: false
```

- Model profile `faster_rcnn_r50_fpn.yml`

```yaml
# Model structure type
architecture: FasterRCNN
# Pretrain model address
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
  # norm_type, Configurable parameter: bn or sync_bn
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
  # The parameters of the proposal are generated during training
  train_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 2000
    post_nms_top_n: 1000
    topk_after_collect: True
  # The parameters of the proposal are generated during evaluation
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
  # Background the threshold
  bg_thresh: 0.5
  # Prospects for threshold
  fg_thresh: 0.5
  # Prospects of proportion
  fg_fraction: 0.25
  # Random sampling
  use_random: True

# TwoFCHead
TwoFCHead:
  # TwoFCHead feature dimension
  out_channel: 1024


# BBoxPostProcess
BBoxPostProcess:
  # decode
  decode: RCNNBox
  # nms
  nms:
    # use MultiClassNMS
    name: MultiClassNMS
    keep_top_k: 100
    score_threshold: 0.05
    nms_threshold: 0.5

```

- runtime configuration file `runtime.yml`

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
