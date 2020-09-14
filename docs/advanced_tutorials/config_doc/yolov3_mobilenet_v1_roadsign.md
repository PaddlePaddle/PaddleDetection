
```yml
#####################################基础配置#####################################
# 检测算法使用YOLOv3，backbone使用MobileNet_v1，数据集使用roadsign_voc的配置文件模板，本配置文件默认使用单卡
# 检测模型的名称
architecture: YOLOv3
# 根据硬件选择是否使用GPU
use_gpu: true
  # ### max_iters为最大迭代次数，而一个iter会运行batch_size * device_num张图片。batch_size在下面 TrainReader.batch_size设置。
max_iters: 1200
# log平滑参数，平滑窗口大小，会从取历史窗口中取log_smooth_window大小的loss求平均值
log_smooth_window: 20
# 模型保存文件夹
save_dir: output
# 每隔多少迭代保存模型
snapshot_iter: 200
# ### mAP 评估方式，mAP评估方式可以选择COCO和VOC或WIDERFACE，其中VOC有11point和integral两种评估方法
metric: COCO
# ### pretrain_weights 可以是imagenet的预训练好的分类模型权重，也可以是在VOC或COCO数据集上的预训练的检测模型权重
# 模型配置文件和权重文件可参考[模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md)
pretrain_weights: https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
# 模型保存文件夹，如果开启了--eval，会在这个文件夹下保存best_model
weights: output/yolov3_mobilenet_v1_roadsign_coco_template/
# ### 根据用户数据设置类别数，注意这里不含背景类
num_classes: 4
# finetune时忽略的参数，按照正则化匹配，匹配上的参数会被忽略掉
finetune_exclude_pretrained_params: ['yolo_output']
# use_fine_grained_loss
use_fine_grained_loss: false

# 检测模型的结构
YOLOv3:
  # 默认是 MobileNetv1
  backbone: MobileNet
  yolo_head: YOLOv3Head

# 检测模型的backbone
MobileNet:
  norm_decay: 0.
  conv_group_scale: 1
  with_extra_blocks: false

# 检测模型的Head
YOLOv3Head:
  # anchor_masks
  anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  # 3x3 anchors
  anchors: [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]]
  # yolo_loss
  yolo_loss: YOLOv3Loss
  # nms 类型参数，可以设置为[MultiClassNMS, MultiClassSoftNMS, MatrixNMS], 默认使用 MultiClassNMS
  nms:
    # background_label，背景标签（类别）的索引，如果设置为 0 ，则忽略背景标签（类别）。如果设置为 -1 ，则考虑所有类别。默认值：0
    background_label: -1
    # NMS步骤后每个图像要保留的总bbox数。 -1表示在NMS步骤之后保留所有bbox。
    keep_top_k: 100
    # 在NMS中用于剔除检测框IOU的阈值，默认值：0.3 。
    nms_threshold: 0.45
    # 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数。
    nms_top_k: 1000
    # 是否归一化，默认值：True 。
    normalized: false
    #  过滤掉低置信度分数的边界框的阈值。
    score_threshold: 0.01

YOLOv3Loss:
  # 这里的batch_size与训练中的batch_size（即TrainReader.batch_size）不同.
  # 仅且当use_fine_grained_loss=true时，计算Loss时使用，且必须要与TrainReader.batch_size设置成一样
  batch_size: 8
  # 忽略样本的阈值 ignore_thresh
  ignore_thresh: 0.7
  # 是否使用label_smooth
  label_smooth: true

LearningRate:
  # ### 学习率设置 参考 https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/FAQ.md#faq%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98
  # base_lr
  base_lr: 0.0001
  # 学习率调整策略
  # 具体实现参考[API](fluid.layers.piecewise_decay)
  schedulers:
  # 学习率调整策略
  - !PiecewiseDecay
    gamma: 0.1
    milestones:
    # ### 参考 https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/FAQ.md#faq%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98
    # ### 8/12 11/12
    - 800
    - 1100
  # 在训练开始时，调低学习率为base_lr * start_factor，然后逐步增长到base_lr，这个过程叫学习率热身，按照以下公式更新学习率
  # linear_step = end_lr - start_lr
  # lr = start_lr + linear_step * (global_step / warmup_steps)
  # 具体实现参考[API](fluid.layers.linear_lr_warmup)
  - !LinearWarmup
    start_factor: 0.3333333333333333
    steps: 100

OptimizerBuilder:
  # 默认使用SGD+Momentum进行训练
  # 具体实现参考[API](fluid.optimizer)
  optimizer:
    momentum: 0.9
    type: Momentum
  # 默认使用SGD+Momentum进行训练
  # 具体实现参考[API](fluid.optimizer)
  regularizer:
    factor: 0.0005
    type: L2

#####################################数据配置#####################################

# 模型训练集设置参考
# 训练、验证、测试使用的数据配置主要区别在数据路径、模型输入、数据增强参数设置
# 如果使用 yolov3_reader.yml，下面的参数设置优先级高，会覆盖yolov3_reader.yml中的参数设置，对于用自定义数据建议将数据配置文件写到下面。
# _READER_: 'yolov3_reader.yml'

TrainReader:
  # 训练过程中模型的输入设置
  # 包括图片，图片长宽高等基本信息，图片id，标记的目标框，类别等信息
  # 不同算法，不同数据集 inputs_def 不同，有的算法需要限制输入图像尺寸，有的不需要###
  inputs_def:
    # YOLO 输入图像大小，必须是32的整数倍###
    # 注意需要与下面的图像尺寸的设置保存一致###
    image_shape: [3, 608, 608]
    # 不同算法，不同数据集 fields 不同###
    # YOLO系列 VOC格式数据： ['image', 'gt_bbox', 'gt_class', 'gt_score']，且需要设置num_max_boxes
    # YOLO系列 COCO格式数据：['image', 'gt_bbox', 'gt_class', 'gt_score']，且需要设置num_max_boxes
    # num_max_boxes，每个样本的groud truth的最多保留个数，若不够用0填充。
    num_max_boxes: 50

  # 训练数据集路径
  dataset:
    # 指定数据集格式
    !COCODataSet
    #dataset/xxx/
    #├── annotations
    #│   ├── train.json
    #│   ├── valid.json
    #├── images
    #│   ├── xxx1.png
    #│   ├── xxx2.png
    #│   ├── xxx3.png
    #│   |   ...

    # 数据集相对路径
    dataset_dir: dataset/roadsign_coco
    # 图片文件夹相对路径，路径是相对于dataset_dir的
    image_dir: images
    # anno_path，路径是相对于dataset_dir的
    anno_path: annotations/train.json

    # 对于VOC、COCO等比赛数据集，可以不指定类别标签文件，use_default_label可以是true。
    # 对于用户自定义数据，如果是VOC格式数据，use_default_label必须要设置成false，且需要提供label_list.txt。如果是COCO格式数据，不需要设置这个参数。
    # use_default_label: false

    # 是否包含背景类，若with_background=true，num_classes需要+1
    # YOLO 系列with_background必须是false，FasterRCNN系列是true ###
    with_background: false


  # 1个GPU的batch size，默认为1。需要注意：每个iter迭代会运行batch_size * device_num张图片
  batch_size: 8
  # 共享内存bufsize。注意，缓存是以batch为单位，缓存的样本数据总量为batch_size * bufsize，所以请注意不要设置太大，请根据您的硬件设置。
  bufsize: 2
  # 选择是否打乱所有样本的顺序
  shuffle: true
  # drop_empty 建议设置为true
  drop_empty: true
  # drop_last 如果最后一个batch的图片数量为奇数，选择是否丢掉这个batch不进行训练。
  # 注意，在某些情况下，drop_last=false时训练过程中可能会出错，建议训练时都设置为true
  drop_last: true
  # mixup，-1表示不做Mixup数据增强。注意，这里是epoch为单位，不是iteration
  mixup_epoch: -1
  # 选择是否使用多进程，默认为false
  use_process: false
  # 若选用多进程，设置使用多进程/线程的数目，默认为4，建议与CPU核数一致
  # 开启多进程后，占用内存会成倍增加，根据内存设置###
  worker_num: 4


  # 数据预处理和数据增强部分，此部分设置要特别注意###
  # 不同算法对数据的预处理流程不同，建议使用对应算法默认的数据处理流程。
  # 比如，YOLO、FPN算法，要求输入图像尺寸必须是32的整数倍

  # 以下是对一个batch中的每单张图片做的数据增强
  sample_transforms:
  # 读取Image图像为numpy数组
  # 可以选择将图片从BGR转到RGB，可以选择对一个batch中的图片做mixup增强
  - !DecodeImage
    to_rgb: true
    with_mixup: true
  # MixupImage
  - !MixupImage
    alpha: 1.5
    beta: 1.5
  # ColorDistort
  - !ColorDistort {}
  - !RandomExpand
    fill_value: [123.675, 116.28, 103.53]
    ratio: 1.5
  # box 坐标归一化，仅仅YOLO系列算法需要
  - !NormalizeBox {}
  # 以prob概率随机反转
  - !RandomFlipImage
    is_normalized: true
    prob: 0.5
  # 归一化
  - !NormalizeImage
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    is_scale: true
    is_channel_first: false
  # 如果 bboxes 数量小于 num_max_boxes，填充值为0的 box，仅仅YOLO系列算法需要
  - !PadBox
    num_max_boxes: 50
  # 坐标格式转化，从XYXY转成XYWH，仅仅YOLO系列算法需要
  - !BboxXYXY2XYWH {}

  # 以下是对一个batch中的所有图片同时做的数据增强
  batch_transforms:
  # 多尺度训练时，从list中随机选择一个尺寸，对一个batch数据同时同时resize
  - !RandomShape
    sizes: [608]
  # channel_first
  - !Permute
    channel_first: true
    to_bgr: false


EvalReader:
  # 评估过程中模型的输入设置
  # 1个GPU的batch size，默认为1。需要注意：每个iter迭代会运行batch_size * device_num张图片
  batch_size: 1
  # 共享内存bufsize，共享内存中训练样本数量是： bufsize * batch_size * 2 张图
  bufsize: 1
  # shuffle=false
  shuffle: false
  # 一般的评估时，batch_size=1，drop_empty可设置成 false
  drop_empty: false
  # 一般的评估时，batch_size=1，drop_last可设置成 false
  drop_last: false
  # 选择是否使用多进程，默认为false
  use_process: false
  # 若选用多进程，设置使用多进程/线程的数目，默认为4，建议与CPU核数一致
  # 开启多进程后，占用内存会成倍增加，根据内存设置 ###
  worker_num: 1

  inputs_def:
    # 图像尺寸与上保持一致
    image_shape: [3, 608, 608]
    # fields 字段
    fields: ['image', 'im_size', 'im_id', 'gt_bbox', 'gt_class']
    num_max_boxes: 50

  # 评估数据集路径
  dataset:
    # 指定数据集格式
    !COCODataSet
    #dataset/xxx/
    #├── annotations
    #│   ├── train.json
    #│   ├── valid.json
    #├── images
    #│   ├── xxx1.png
    #│   ├── xxx2.png
    #│   ├── xxx3.png
    #│   |   ...

    # 数据集相对路径
    dataset_dir: dataset/roadsign_coco
    # 图片文件夹相对路径，路径是相对于dataset_dir的
    image_dir: images
    # anno_path，路径是相对于dataset_dir的
    anno_path: annotations/valid.json

    # 对于VOC、COCO等比赛数据集，可以不指定类别标签文件，use_default_label可以是true。
    # 对于用户自定义数据，如果是VOC格式数据，use_default_label必须要设置成false，且需要提供label_list.txt。如果是COCO格式数据，不需要设置这个参数。
    # use_default_label: false

    # 是否包含背景类，若with_background=true，num_classes需要+1
    # YOLO 系列with_background必须是false，FasterRCNN系列是true ###
    with_background: false

  # 单张图的 transforms
  sample_transforms:
    # DecodeImage
    - !DecodeImage
      to_rgb: true

    # 与上面图像尺寸保持一致 ###
    - !ResizeImage
      target_size: 608
      interp: 2
    # 图像归一化
    - !NormalizeImage
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      is_scale: true
      is_channel_first: false
    # 如果 bboxes 数量小于 num_max_boxes，填充值为0的 box
    - !PadBox
      num_max_boxes: 50
    - !Permute
      to_bgr: false
      channel_first: true

TestReader:
  # 测试过程中模型的输入设置
  # 预测时 batch_size设置为1
  batch_size: 1
  # 一般的预测时，batch_size=1，drop_empty可设置成 false
  drop_empty: false
  # 一般的预测时，batch_size=1，drop_last可设置成 false
  drop_last: false


  inputs_def:
    # 预测时输入图像尺寸，与上面图像尺寸保持一致
    image_shape: [3, 608, 608]
    # 预测时需要读取字段
    # fields 字段
    fields: ['image', 'im_size', 'im_id']

  dataset:
    # 预测数据
    !ImageFolder
    # anno_path，路径是相对于dataset_dir的
    anno_path: annotations/valid.json

    # 对于VOC、COCO等比赛数据集，可以不指定类别标签文件，use_default_label可以是true。
    # 对于用户自定义数据，如果是VOC格式数据，use_default_label必须要设置成false，且需要提供label_list.txt。如果是COCO格式数据，不需要设置这个参数。
    # use_default_label: false

    # 是否包含背景类，若with_background=true，num_classes需要+1
    # YOLO 系列with_background必须是false，FasterRCNN系列是true ###
    with_background: false


  # 单张图的 transforms
  sample_transforms:
    # DecodeImage
    - !DecodeImage
      to_rgb: true
    # 注意与上面图像尺寸保持一致
    - !ResizeImage
      target_size: 608
      interp: 2
    # NormalizeImage
    - !NormalizeImage
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      is_scale: true
      is_channel_first: false
    # Permute
    - !Permute
      to_bgr: false
      channel_first: true

```
