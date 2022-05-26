# RCNN系列模型参数配置教程

标签： 模型参数配置

---
```yaml

#####################################基础配置#####################################

# 检测模型的名称
architecture: MaskRCNN
# 默认使用GPU运行，设为False时使用CPU运行
use_gpu: true
# 最大迭代次数，而一个iter会运行batch_size * device_num张图片
# 一般batch_size为1时，1x迭代18万次，2x迭代36万次
max_iters: 180000
# 模型保存间隔，如果训练时eval设置为True，会在保存后进行验证
snapshot_iter: 10000
# 输出指定区间的平均结果，默认20，即输出20次的平均结果。也是默认打印log的间隔。
log_iter: 20
# 训练权重的保存路径
save_dir: output
# 模型的预训练权重，默认是从指定url下载
pretrain_weights: https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar
# 验证模型的评测标准，可以选择COCO或者VOC
metric: COCO
# 用于模型验证或测试的训练好的权重
weights: output/mask_rcnn_r50_fpn_1x/model_final/
# 用于训练或验证的数据集的类别数目
# **其中包含背景类，即81=80 + 1（背景类）**
num_classes: 81

#####################################模型配置#####################################

# Mask RCNN元结构，包括了以下主要组件, 具体细节可以参考[论文]( https://arxiv.org/abs/1703.06870)
MaskRCNN:
  backbone: ResNet
  fpn: FPN
  rpn_head: FPNRPNHead
  roi_extractor: FPNRoIAlign
  bbox_assigner: BBoxAssigner
  bbox_head: BBoxHead
  mask_assigner: MaskAssigner
  mask_head: MaskHead
  rpn_only: false

# 主干网络
ResNet:
  # 配置在哪些阶段加入可变性卷积，默认不添加
  dcn_v2_stages: []
  # ResNet深度，默认50
  depth: 50
  # 主干网络返回的主要阶段特征用于FPN作进一步的特征融合
  # 默认从[2,3,4,5]返回特征
  feature_maps: [2,3,4,5]
  # 是否在训练中固定某些权重，默认从第2阶段开始固定，即resnet的stage 1
  freeze_at: 2
  # 是否停止norm layer的梯度回传，默认是
  freeze_norm: true
  # norm layer的权重衰退值
  norm_decay: 0.0
  # norm layer的类型, 可以选择bn/sync_bn/affine_channel, 默认为affine_channel
  norm_type: affine_channel
  # ResNet模型的类型, 分为'a', 'b', 'c', 'd'四种, 默认使用'b'类型
  variant: b

# FPN多特征融合
FPN:
  # FPN使用的最高层特征后是否添加额外conv，默认false
  has_extra_convs: false
  # FPN使用主干网络最高层特征，默认是resnet第5阶段后添加额外卷积操作变<成了FPN的第6个，总共有5个阶段
  max_level: 6
  # FPN使用主干网络最低层特征，默认是resnet第2阶段的输出
  min_level: 2
  # FPN中使用Norm类型, bn/sync_bn/affine_channel/null, 默认不用null
  norm_type: null
  # FPN输出特征的通道数量, 默认是256
  num_chan: 256
  # 特征图缩放比例, 默认是[0.03125, 0.0625, 0.125, 0.25]
  spatial_scale: [0.03125, 0.0625, 0.125, 0.25]

# 检测第一阶段RPN
FPNRPNHead:
  # FPN第一层特征生成anchor时，默认anchor尺寸32
  anchor_start_size: 32
  # RPNHead默认进行前背景二分类
  num_classes: 1
  # 根据特征图尺寸，在特征图的每个位置生成N个大小、长宽比各不同anchor
  # N = anchor_sizes * aspect_ratios
  # 具体实现参考[API](fluid.layers.anchor_generator)
  anchor_generator:
    aspect_ratios: [0.5, 1.0, 2.0]
    variance: [1.0, 1.0, 1.0, 1.0]
  # 首先计算Anchor和GT BBox之间的IoU，为每个Anchor匹配上GT，
  # 然后根据阈值过滤掉IoU低的Anchor，得到最终的Anchor及其GT进行loss计算
  # 具体实现参考[API](fluid.layers.rpn_target_assign)
  rpn_target_assign:
    rpn_batch_size_per_im: 256
    rpn_fg_fraction: 0.5
    rpn_negative_overlap: 0.3
    rpn_positive_overlap: 0.7
    rpn_straddle_thresh: 0.0
  # 首先取topk个分类分数高的anchor
  # 然后通过NMS对这topk个anchor进行重叠度检测，对重叠高的两个anchor只保留得分高的
  # 训练和测试阶段主要区别在最后NMS保留的Anchor数目
  # 训练时输出2000个proposals，推理时输出1000个proposals
  # 具体实现参考[API](fluid.layers.generate_proposals)
  train_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    post_nms_top_n: 2000
    pre_nms_top_n: 2000
  test_proposal:
    min_size: 0.0
    nms_thresh: 0.7
    post_nms_top_n: 1000
    pre_nms_top_n: 1000

# 对FPN每层执行RoIAlign后，然后合并输出结果，用于BBox Head计算
FPNRoIAlign:
  # 用于抽取特征特征的FPN的层数，默认为4
  canconical_level: 4
  # 用于抽取特征特征的FPN的特征图尺寸，默认为224
  canonical_size: 224
  # 用于抽取特征特征的最高层FPN，默认是2
  max_level: 5
  # 用于抽取特征特征的最底层FPN，默认是2
  min_level: 2
  #roi extractor的采样率，默认为2
  sampling_ratio: 2
  # 输出bbox的特征图尺寸，默认为7
  box_resolution: 7
  # 输出mask的特征图尺寸，默认为14
  mask_resolution: 14

# 输出实例掩码的Head
MaskHead:
  # 卷积的数量，FPN是4，其他为0，默认为0
  num_convs: 4
  # mask head输出的特征图尺寸，默认14
  resolution: 28
  # 空洞率，默认为1
  dilation: 1
  # 第一个卷积后输出的特征图通道数, 默认为256
  num_chan_reduced: 256
  # 输出的mask的类别，默认为81
  num_classes: 81

# 求rpn生成的roi跟gt bbox之间的iou，然后根据阈值进行过滤，保留一定数量的roi
# 再根据gt bbox的标签，对roi进行标签赋值，即得到每个roi的类别
# 具体实现参考[API](fluid.layers.generate_proposal_labels)
BBoxAssigner:
  batch_size_per_im: 512
  bbox_reg_weights: [0.1, 0.1, 0.2, 0.2]
  bg_thresh_lo: 0.0
  bg_thresh_hi: 0.5
  fg_fraction: 0.25
  fg_thresh: 0.5

# 根据roi的label，选择前景，为其赋值mask label
# 具体实现参考[API](fluid.layers.generate_mask_labels)
MaskAssigner:
  resolution: 28
  num_classes: 81

# 输出检测框的Head
BBoxHead:
  # 在roi extractor和bbox head之间，插入两层FC，对特征做进一步学习
  head: TwoFCHead
  # 通过NMS进行bbox过滤
  # 具体实现参考[API](fluid.layers.multiclass_nms)
  nms:
    keep_top_k: 100
    nms_threshold: 0.5
    score_threshold: 0.05

# 输出检测框之前，对特征进一步学习
TwoFCHead:
  # FC输出的特征图通道数，默认是1024
  num_chan: 1024

#####################################训练配置#####################################

# 学习率配置
LearningRate:
  # 初始学习率, 一般情况下8卡gpu，batch size为2时设置为0.02
  # 可以根据具体情况，按比例调整
  # 比如说4卡V100，bs=2时，设置为0.01
  base_lr: 0.01
  # 学习率规划器
  # 具体实现参考[API](fluid.layers.piecewise_decay)
  schedulers:
    # 学习率衰减策略
    # 对于coco数据集，1个epoch大概需要7000个iter
    # if step < 120000:
    #    learning_rate = 0.1
    # elif 120000 <= step < 160000:
    #    learning_rate = 0.1 * 0.1
    # else:
    #    learning_rate = 0.1 * (0.1)**2
    - !PiecewiseDecay
      gamma: 0.1
      milestones: [120000, 160000]
    # 在训练开始时，调低学习率为base_lr * start_factor，然后逐步增长到base_lr，这个过程叫学习率热身，按照以下公式更新学习率
    # linear_step = end_lr - start_lr
    # lr = start_lr + linear_step * (global_step / warmup_steps)
    # 具体实现参考[API](fluid.layers.linear_lr_warmup)
    - !LinearWarmup
      start_factor: 0.3333333333333333
      steps: 500

OptimizerBuilder:
  # 默认使用SGD+Momentum进行训练
  # 具体实现参考[API](fluid.optimizer)
  optimizer:
    momentum: 0.9
    type: Momentum
  # 默认使用L2权重衰减正则化
  # 具体实现参考[API](fluid.regularizer)
  regularizer:
    factor: 0.0001
    type: L2

#####################################数据配置#####################################

# 模型训练集设置参考
# 训练、验证、测试使用的数据配置主要区别在数据路径、模型输入、数据增强参数设置
TrainReader:
  # 训练过程中模型的相关输入
  # 包括图片，图片长宽高等基本信息，图片id，标记的目标框、实例标签、实例分割掩码
  inputs_def:
    fields: ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd', 'gt_mask']
  # VOC数据集对应的输入，注意选择VOC时，也要对应修改metric: VOC
- # fields: ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_difficult']
  # 数据集目录配置
  dataset:
    # 指定数据集名称，可以选择VOCDataSet, COCODataSet
    !COCODataSet
    # 训练图片所在目录
    image_dir: train2017
    # 标记文件所在目录
    anno_path: annotations/instances_train2017.json
    # 数据集根目录
    dataset_dir: dataset/coco
  # 对一个batch中的单张图片做的数据增强
  sample_transforms:
  # 读取Image图像为numpy数组
  # 可以选择将图片从BGR转到RGB，可以选择对一个batch中的图片做mixup增强
  - !DecodeImage
    to_rgb: true
  # 对图片进行随机翻转
  # 可以选择同步翻转mask，可以选择归一化bbox的坐标
  - !RandomFlipImage
    prob: 0.5
  # 归一化图片，默认均值[0.485, 0.456, 0.406]，方差[1, 1, 1]
  # 可以选择将归一化结果除以255，可以选择图片的数据格式
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.485,0.456,0.406]
    std: [0.229, 0.224,0.225]
  # 调整图片尺寸，默认采用cv2的线性插值
  - !ResizeImage
    target_size: 800
    max_size: 1333
    interp: 1
    use_cv2: true
  # 调整图片数据格式，默认使用CHW
  - !Permute
    to_bgr: false
    channel_first: true
  # 对一个batch中的图片统一做的数据增强
  batch_transforms:
  # 将一个batch中的图片，按照最大的尺寸，做补齐
  - !PadBatch
    pad_to_stride: 32
    # 选择是否使用padding之后的image信息，默认为false
    use_padded_im_info: false
  # 1个GPU的batch size，默认为1
  batch_size: 1
  # 选择是否打乱所有样本的顺序
  shuffle: true
  # 使用多进程/线程的数目，默认为2
  worker_num: 2
  # 选择是否使用多进程，默认为false
  use_process: false
  # 如果最后一个batch的图片数量为奇数，选择是否丢掉这个batch，不进行训练，默认是不丢掉的
  drop_last: false
  # 使用数据集中的样本数目，默认是-1，表示使用全部
  samples: -1

  # 模型验证集设置参考
  EvalReader:
  # 验证过程中模型的相关输入
  # 包括图片，图片长宽高等基本信息，图片id，图片shape
  inputs_def:
    fields: ['image', 'im_info', 'im_id', 'im_shape']
  dataset:
    !COCODataSet
    image_dir: val2017
    anno_path: annotations/instances_val2017.json
    dataset_dir: dataset/coco
  sample_transforms:
  - !DecodeImage
    to_rgb: true
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.485,0.456,0.406]
    std: [0.229, 0.224,0.225]
  - !ResizeImage
    interp: 1
    max_size: 1333
    target_size: 800
    use_cv2: true
  - !Permute
    channel_first: true
    to_bgr: false
  batch_size: 1
  shuffle: false
  drop_last: false
  drop_empty: false
  worker_num: 2

# 测试验证集设置参考
TestReader:
  # 测试过程中模型的相关输入
  # 包括图片，图片长宽高等基本信息，图片id，图片shape
  inputs_def:
    fields: ['image', 'im_info', 'im_id', 'im_shape']
  dataset:
    # 测试图片所在目录
    !ImageFolder
    anno_path: annotations/instances_val2017.json
  sample_transforms:
  - !DecodeImage
    to_rgb: true
    with_mixup: false
  - !NormalizeImage
    is_channel_first: false
    is_scale: true
    mean: [0.485,0.456,0.406]
    std: [0.229, 0.224,0.225]
  - !ResizeImage
    interp: 1
    max_size: 1333
    target_size: 800
    use_cv2: true
  - !Permute
    channel_first: true
    to_bgr: false
  batch_size: 1
  shuffle: false
  drop_last: false
  ```
