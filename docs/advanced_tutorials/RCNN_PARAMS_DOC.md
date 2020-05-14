# RCNN系列模型参数配置教程

标签： 模型参数配置

---

\# 检测模型的名称<br>
architecture: MaskRCNN<br>

\# 默认使用GPU运行，设为False时使用CPU运行<br>
use_gpu: true<br>

\# 最大迭代次数，而一个iter会运行batch_size * device_num张图片<br>
\# 一般batch_size为1时，1x迭代18万次，2x迭代36万次<br>
max_iters: 180000<br>

\# 模型保存间隔，如果训练时eval设置为True，会在保存后进行验证<br>
snapshot_iter: 10000<br>

\# 输出指定区间的平均结果，默认20，即输出20次的平均结果<br>
log_smooth_window: 20<br>

\# 默认打印log的间隔，默认20<br>
log_iter: 20<br>

\# 训练权重的保存路径<br>
save_dir: output<br>

\# 模型的预训练权重，默认是从指定url下载<br>
pretrain_weights: https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar<br>

\# 验证模型的评测标准，可以选择COCO或者VOC<br>
metric: COCO<br>

\# 用于模型验证或测试的训练好的权重<br>
weights: output/mask_rcnn_r50_fpn_1x/model_final/<br>

\# 用于训练或验证的数据集的类别数目<br>
\# **其中包含背景类，即81=80 + 1（背景类）**<br>
num_classes: 81<br>

Mask RCNN元结构，包括了以下主要组件, 具体细节可以参考[论文]( https://arxiv.org/abs/1703.06870)<br>

MaskRCNN:<br>
  
  >backbone: ResNet <br>
  >fpn: FPN<br>
  >roi_extractor: FPNRoIAlign<br>
  >rpn_head: FPNRPNHead<br>
  >bbox_assigner: BBoxAssigner<br>
  >bbox_head: BBoxHead<br>
  >mask_assigner: MaskAssigner<br>
  >mask_head: MaskHead<br>
  >rpn_only: false<br>

主干网络<br>
ResNet:<br>
  >\# 配置在哪些阶段加入可变性卷积，默认不添加<br>
  >dcn_v2_stages: []<br>
  >\# ResNet深度，默认50<br>
  >depth: 50<br>
  >\# 主干网络返回的主要阶段特征用于FPN作进一步的特征融合<br>
  >\# 默认从[2,3,4,5]返回特征<br>
  >feature_maps:<br>
  >>\- 2<br>
  >>\- 3<br>
  >>\- 4<br>
  >>\- 5<br>
  
  >\# 是否在训练中固定norm layer的权重，默认从第2阶段开始固定<br>
  >freeze_at: 2<br>
  >\# 是否停止norm layer的梯度回传，默认是<br>
  >freeze_norm: true<br>
  >\# norm layer的权重衰退值<br>
  >norm_decay: 0.0<br>
  >\# norm layer的类型, 可以选择bn/sync_bn/affine_channel, 默认为affine_channel<br>
  >norm_type: affine_channel<br>
  >\# ResNet模型的类型, 分为'a', 'b', 'c', 'd'四种, 默认使用'b'类型<br>
  >variant: b<br>

FPN多特征融合<br>
FPN:<br>
  >\# FPN使用的最高层特征后是否添加额外conv，默认false<br>
  has_extra_convs: false<br>
  >\# FPN使用主干网络最高层特征，默认是resnet第5阶段后添加额外卷积操作变<成了FPN的第6个，总共有5个阶段<br>
  max_level: 6<br>
  >\# FPN使用主干网络最低层特征，默认是resnet第2阶段的输出<br>
  min_level: 2<br>
  >\# FPN中使用Norm类型, bn/sync_bn/affine_channel/null, 默认不用null<br>
  norm_type: null<br>
  >\# FPN输出特征的通道数量, 默认是256<br>
  num_chan: 256<br>
  >\# 特征图缩放比例, 默认是[0.03125, 0.0625, 0.125, 0.25]<br>
  spatial_scale:<br>
  >>\- 0.03125<br>
  >>\- 0.0625<br>
  >>\- 0.125<br>
  >>\- 0.25<br>

检测第一阶段RPN<br>
FPNRPNHead:<br>
  >\# FPN第一层特征生成anchor时，默认anchor尺寸32<br>
  >anchor_start_size: 32<br>
  >\# RPNHead默认进行前背景二分类<br>
  >num_classes: 1<br>
  >\# 根据特征图尺寸，在特征图的每个位置生成N个大小、长宽比各不同anchor<br>
  >\# N = anchor_sizes * aspect_ratios<br>
  >\# 具体实现参考[API](fluid.layers.anchor_generator)<br>
  >anchor_generator:<br>
    >>aspect_ratios:<br>
    >>\- 0.5<br>
    >>\- 1.0<br>
    >>\- 2.0<br>
    >>variance:<br>
    >>\- 1.0<br>
    >>\- 1.0<br>
    >>\- 1.0<br>
    >>\- 1.0<br>
    
  >\# 首先计算Anchor和GT BBox之间的IoU，为每个Anchor匹配上GT，<br>
  >\# 然后根据阈值过滤掉IoU低的Anchor，得到最终的Anchor及其GT进行loss计算<br>
  >\# 具体实现参考[API](fluid.layers.rpn_target_assign)<br>
  >rpn_target_assign:<br>
    >>rpn_batch_size_per_im: 256<br>
    >>rpn_fg_fraction: 0.5<br>
    >>rpn_negative_overlap: 0.3<br>
    >>rpn_positive_overlap: 0.7<br>
    >>rpn_straddle_thresh: 0.0<br>
  
  >\# 首先取topk个分类分数高的anchor，<br>
  >\# 然后通过NMS对这topk个anchor进行重叠度检测，对重叠高的两个anchor只保留得分高的。<br>
  >\# 训练和测试阶段主要区别在最后NMS保留的Anchor数目。<br>
  >\# 具体实现参考[API](fluid.layers.generate_proposals)<br>
  >train_proposal:<br>
    >>min_size: 0.0<br>
    >>nms_thresh: 0.7<br>
    >>post_nms_top_n: 2000<br>
    >>pre_nms_top_n: 2000<br>

  >test_proposal:<br>
    >>min_size: 0.0<br>
    >>nms_thresh: 0.7<br>
    >>post_nms_top_n: 1000<br>
    >>pre_nms_top_n: 1000<br>

对FPN每层执行RoIAlign后，然后合并输出结果，用于BBox Head计算<br>
FPNRoIAlign:<br>
  >\# 用于抽取特征特征的FPN的层数，默认为4<br>
  >canconical_level: 4<br>
  >\# 用于抽取特征特征的FPN的特征图尺寸，默认为224<br>
  >canonical_size: 224<br>
  >\# 用于抽取特征特征的最高层FPN，默认是2<br>
  >max_level: 5<br>
  >\# 用于抽取特征特征的最底层FPN，默认是2<br>
  >min_level: 2<br>
  >\#roi extractor的采样率，默认为2<br>
  >sampling_ratio: 2<br>
  >\# 输出bbox的特征图尺寸，默认为7<br>
  >box_resolution: 7<br>
  >\# 输出mask的特征图尺寸，默认为14<br>
  >mask_resolution: 14<br>

输出实例掩码的Head<br>
MaskHead:<br>
  >\# 卷积的数量，FPN是4，其他为0，默认为0<br>
  >num_convs: 4<br>
  >\# mask head输出的特征图尺寸，默认14<br>
  >resolution: 28<br>
  >\# 空洞率，默认为1<br>
  >dilation: 1<br>
  >\# 第一个卷积后输出的特征图通道数, 默认为256<br>
  >num_chan_reduced: 256<br>
  >\# 输出的mask的类别，默认为81<br>
  >num_classes: 81<br>

>\# 求rpn生成的roi跟gt bbox之间的iou，然后根据阈值进行过滤，保留一定数量的roi<br>
>\# 再根据gt bbox的标签，对roi进行标签赋值，即得到每个roi的类别<br>
>\# 具体实现参考[API](fluid.layers.generate_proposal_labels)<br>
>BBoxAssigner:<br>
  >>batch_size_per_im: 512<br>
  >>bbox_reg_weights:<br>
  >>bg_thresh_hi: 0.5<br>
  >>bg_thresh_lo: 0.0<br>
  >>fg_fraction: 0.25<br>
  >>fg_thresh: 0.5<br>
  >>num_classes: 81<br>
  >>shuffle_before_sample: true<br>
  >>>\- 0.1<br>
  >>>\- 0.1<br>
  >>>\- 0.2<br>
  >>>\- 0.2<br>

>\# 根据roi的label，选择前景，为其赋值mask label<br>
>\# 具体实现参考[API](fluid.layers.generate_mask_labels)<br>
>MaskAssigner:<br>
  >>resolution: 28<br>
  >>num_classes: 81<br>

输出检测框的Head<br>
BBoxHead:<br>
  >\# 在roi extractor和bbox head之间，插入两层FC，对特征做进一步学习<br>
  >head: TwoFCHead<br>
  >>\# 通过NMS进行bbox过滤<br>
  >>\# 具体实现参考[API](fluid.layers.multiclass_nms)<br>
  >>keep_top_k: 100<br>
  >>nms_threshold: 0.5<br>
  >>score_threshold: 0.05<br>
  >>num_classes: 81<br>
  >>\# 对bbox的坐标进行编解码操作<br>
  >>\# 具体实现参考[API](fluid.layers.box_coder)<br>
  >>box_coder:<br>
    >>>axis: 1<br>
    >>>box_normalized: false<br>
    >>>code_type: decode_center_size<br>
    >>>prior_box_var:<br>
    >>>>\- 0.1<br>
    >>>>\- 0.1<br>
    >>>>\- 0.2<br>
    >>>>\- 0.2<br>

输出检测框之前，对特征进一步学习<br>
TwoFCHead:<br>
  >\# FC输出的特征图通道数，默认是1024<br>
  >num_chan: 1024<br>

学习率配置<br>
LearningRate:<br>
  >\# 初始学习率, 一般情况下8卡gpu，batch size为2时设置为0.02<br>
  >\# 可以根据具体情况，按比例调整<br>
  >\# 比如说4卡V100，bs=2时，设置为0.01<br>
  >base_lr: 0.01<br>
  
  >\# 学习率规划器<br>
  >\# 具体实现参考[API](fluid.layers.piecewise_decay)<br>
  >schedulers:<br>
  >>\- !PiecewiseDecay<br>
    >>>gamma: 0.1<br>
    >>>milestones:<br>
    >>>>\- 120000<br>
    >>>>\- 160000<br>
    values: null<br>
  >>\# 在训练开始时，调低学习率为base_lr * start_factor，然后逐步增长到base_lr，这个过程叫学习率热身，按照以下公式更新学习率<br>
  >>\# linear_step = end_lr - start_lr<br>
  >>\# lr = start_lr + linear_step * (global_step / warmup_steps)<br>
  >>\# 具体实现参考[API](fluid.layers.linear_lr_warmup)<br>
  >>\- !LinearWarmup<br>
    >>>start_factor: 0.3333333333333333<br>
    >>>steps: 500<br>

OptimizerBuilder:<br>
  >\# 默认使用SGD+Momentum进行训练<br>
  >\# 具体实现参考[API](fluid.optimizer)<br>
  >optimizer:<br>
    >>momentum: 0.9<br>
    >>type: Momentum<br>

  >\# 默认使用L2权重衰减正则化<br>
  >\# 具体实现参考[API](fluid.regularizer)<br>
  >regularizer:<br>
    >>factor: 0.0001<br>
    >>type: L2<br>


\# 模型训练集设置参考 <br>
\# 训练、验证、测试使用的数据配置主要区别在数据路径、模型输入、数据增强参数设置<br>
TrainReader:<br>
  >\# 1个GPU的batch size，默认为1<br>
  >batch_size: 1<br>

  >\# 数据集目录配置<br>
  >dataset:<br>
    >>\# 数据集根目录<br>
    >>dataset_dir: dataset/coco<br>
    >>\# 标记文件所在目录<br>
    >>annotation: annotations/instances_train2017.json<br>
    >>\# 训练图片所在目录<br>
    >>image_dir: train2017<br>

  >\# 训练过程中模型的相关输入<br>
  >fields:<br>
  >>\- image<br>
  >>\- im_info<br>
  >>\- im_id<br>
  >>\- gt_box<br>
  >>\- gt_label<br>
  >>\- is_crowd<br>
  >>\- gt_mask<br>
  
  >\# 输入Image的尺寸<br>
  >image_shape:<br>
  >>\- 3<br>
  >>\- 800<br>
  >>\- 1333<br>
  
  >\# 对一个batch中的单张图片做的数据增强<br>
  >sample_transforms:<br>
  >>\# 读取Image图像为numpy数组，<br>
  >>\# 可以选择将图片从BGR转到RGB，可以选择对一个batch中的图片做mixup增强<br>
  >>\- !DecodeImage<br>
    to_rgb: true  # default: true<br>
    with_mixup: false  # default: false<br>
  >>\# 对图片进行随机翻转，<br>
  >>\# 可以选择同步翻转mask，可以选择归一化bbox的坐标<br>
  >>\- !RandomFlipImage<br>
    is_mask_flip: true  # default: false<br>
    is_normalized: false  # default: false<br>
    prob: 0.5  # default: 0.5<br>
  >>\# 归一化图片，默认均值[0.485, 0.456, 0.406]，方差[1, 1, 1]<br>
  >>\# 可以选择将归一化结果除以255，可以选择图片的数据格式<br>
  >>\- !NormalizeImage<br>
    >>is_channel_first: false<br>
    >>is_scale: true<br>
    >>mean:<br>
    \- 0.485<br>
    \- 0.456<br>
    \- 0.406<br>
    std:<br>
    \- 0.229<br>
    \- 0.224<br>
    \- 0.225<br>

  >>\# 调整图片尺寸，默认采用cv2的线性插值<br>
  >>\- !ResizeImage<br>
    >>>interp: 1<br>
    >>>max_size: 1333<br>
    >>>target_size: 800<br>
    >>>use_cv2: true  # default: true<br>

  >>\# 调整图片数据格式，默认使用CHW<br>
  >>\- !Permute<br>
     >>>channel_first: true<br>
     >>>to_bgr: false  # default: true<br>  
  
  >\# 对一个batch中的图片统一做的数据增强<br>
  >batch_transforms:<br>
  >>\# 将一个batch中的图片，按照最大的尺寸，做补齐<br>
  >>\- !PadBatch<br>
    >>>pad_to_stride: 32  # default: 32<br>

  >\# 如果最后一个batch的图片数量为奇数，选择是否丢掉这个batch，不进行训练，默认是不丢掉的<br>
  drop_last: false<br>
  >\# 使用的进程数目，默认为2<br>
  num_workers: 2<br>
  >\# 使用数据集中的样本数目，默认是-1，表示使用全部<br>
  samples: -1<br>
  >\# 选择是否打乱所有样本的顺序<br>
  shuffle: true<br>
  >\# 选择是否更新padding之后的数据信息，默认为false<br>
  use_padded_im_info: false<br>
  >\# 选择是否使用多进程，默认为false<br>
  use_process: false<br>