**# config yaml配置项说明**

KeyPoint 使用时config文件配置项说明，以[tinypose_256x192.yml](../../configs/keypoint/tiny_pose/tinypose_256x192.yml)为例

```yaml
use_gpu: true                                                                                  #是否使用gpu训练

log_iter: 5                                                                                    #打印log的iter间隔

save_dir: output                                                                               #模型保存目录

snapshot_epoch: 10                                                                             #保存模型epoch间隔

weights: output/tinypose_256x192/model_final                                                   #测试加载模型路径（不含后缀“.pdparams”）

epoch: 420                                                                                     #总训练epoch数量

num_joints: &num_joints 17                                                                     #关键点数量

pixel_std: &pixel_std 200                                                                      #变换时相对比率像素（无需关注，不动就行）

metric: KeyPointTopDownCOCOEval                                                                #metric评估函数

num_classes: 1                                                                                 #种类数（检测模型用，不需关注）

train_height: &train_height 256                                                                #模型输入尺度高度变量设置

train_width: &train_width 192                                                                  #模型输入尺度宽度变量设置

trainsize: &trainsize [*train_width, *train_height]                                            #模型输入尺寸，使用已定义变量

hmsize: &hmsize [48, 64]                                                                       #输出热力图尺寸（宽，高）

flip_perm: &flip_perm [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]  #左右关键点经图像翻转时对应关系，例如：图像翻转后，左手腕变成了右手腕，右手腕变成了左手腕





\#####model

architecture: TopDownHRNet                                                                     #模型框架结构类选择



TopDownHRNet:                                                                                  #TopDownHRNet相关配置

  backbone: LiteHRNet                                                                          #模型主干网络

  post_process: HRNetPostProcess                                                               #模型后处理类

  flip_perm: *flip_perm                                                                        #同上flip_perm

  num_joints: *num_joints                                                                      #关键点数量（输出通道数量）

  width: &width 40                                                                             #backbone输出通道数

  loss: KeyPointMSELoss                                                                        #loss函数选择

  use_dark: true                                                                               #是否使用DarkPose后处理



LiteHRNet:                                                                                     #LiteHRNet相关配置

  network_type: wider_naive                                                                    #网络结构类型选择

  freeze_at: -1                                                                                #梯度截断branch id，截断则该branch梯度不会反传

  freeze_norm: false                                                                           #是否固定normalize层参数

  return_idx: [0]                                                                              #返回feature的branch id



KeyPointMSELoss:                                                                               #Loss相关配置

  use_target_weight: true                                                                      #是否使用关键点权重

  loss_scale: 1.0                                                                              #loss比率调整，1.0表示不变



\#####optimizer

LearningRate:                                                                                  #学习率相关配置

  base_lr: 0.002                                                                               #初始基础学习率

  schedulers:

  \- !PiecewiseDecay                                                                           #衰减策略

​    milestones: [380, 410]                                                                     #衰减时间对应epoch次数

​    gamma: 0.1                                                                                 #衰减率

  \- !LinearWarmup                                                                             #Warmup策略

​    start_factor: 0.001                                                                        #warmup初始学习率比率

​    steps: 500                                                                                 #warmup所用iter次数



OptimizerBuilder:                                                                              #学习策略设置

  optimizer:

​    type: Adam                                                                                 #学习策略Adam

  regularizer:

​    factor: 0.0                                                                                #正则项权重

​    type: L2                                                                                   #正则类型L2/L1





\#####data

TrainDataset:                                                                                  #训练数据集设置

  !KeypointTopDownCocoDataset                                                                  #数据加载类

​    image_dir: ""                                                                              #图片文件夹,对应dataset_dir/image_dir

​    anno_path: aic_coco_train_cocoformat.json                                                  #训练数据Json文件，coco格式

​    dataset_dir: dataset                                                                       #训练数据集所在路径，image_dir、anno_path路径基于此目录

​    num_joints: *num_joints                                                                    #关键点数量，使用已定义变量

​    trainsize: *trainsize                                                                      #训练使用尺寸，使用已定义变量

​    pixel_std: *pixel_std                                                                      #同上pixel_std

​    use_gt_bbox: True                                                                          #是否使用gt框





EvalDataset:                                                                                   #评估数据集设置

  !KeypointTopDownCocoDataset                                                                  #数据加载类

​    image_dir: val2017                                                                         #图片文件夹

​    anno_path: annotations/person_keypoints_val2017.json                                       #评估数据Json文件，coco格式

​    dataset_dir: dataset/coco                                                                  #数据集路径，image_dir、anno_path路径基于此目录

​    num_joints: *num_joints                                                                    #关键点数量，使用已定义变量

​    trainsize: *trainsize                                                                      #训练使用尺寸，使用已定义变量

​    pixel_std: *pixel_std                                                                      #同上pixel_std

​    use_gt_bbox: True                                                                          #是否使用gt框，一般测试时用

​    image_thre: 0.5                                                                            #检测框阈值设置，测试时使用非gt_bbox时用



TestDataset:                                                                                   #纯测试数据集设置，无label

  !ImageFolder                                                                                 #数据加载类，图片文件夹类型

​    anno_path: dataset/coco/keypoint_imagelist.txt                                             #测试图片列表文件



worker_num: 2                                                                                  #数据加载worker数量，一般2-4，太多可能堵塞

global_mean: &global_mean [0.485, 0.456, 0.406]                                                #全局均值变量设置

global_std: &global_std [0.229, 0.224, 0.225]                                                  #全局方差变量设置

TrainReader:                                                                                   #训练数据加载类设置

  sample_transforms:                                                                           #数据预处理变换设置

​    \- RandomFlipHalfBodyTransform:                                                            #随机翻转&随机半身变换类

​        scale: 0.25                                                                            #最大缩放尺度比例

​        rot: 30                                                                                #最大旋转角度

​        num_joints_half_body: 8                                                                #关键点小于此数不做半身变换

​        prob_half_body: 0.3                                                                    #半身变换执行概率（满足关键点数量前提下）

​        pixel_std: *pixel_std                                                                  #同上pixel_std

​        trainsize: *trainsize                                                                  #训练尺度，同上trainsize

​        upper_body_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                     #上半身关键点id

​        flip_pairs: *flip_perm                                                                 #左右关键点对应关系，同上flip_perm

​    \- AugmentationbyInformantionDropping:

​        prob_cutout: 0.5                                                                       #随机擦除变换概率

​        offset_factor: 0.05                                                                    #擦除位置中心点随机波动范围相对图片宽度比例

​        num_patch: 1                                                                           #擦除位置数量

​        trainsize: *trainsize                                                                  #同上trainsize

​    \- TopDownAffine:

​        trainsize: *trainsize                                                                  #同上trainsize

​        use_udp: true                                                                          #是否使用udp_unbias（flip测试使用）

​    \- ToHeatmapsTopDown_DARK:                                                                 #生成热力图gt类

​        hmsize: *hmsize                                                                        #热力图尺寸

​        sigma: 2                                                                               #生成高斯核sigma值设置

  batch_transforms:

​    \- NormalizeImage:                                                                         #图像归一化类

​        mean: *global_mean                                                                     #均值设置，使用已有变量

​        std: *global_std                                                                       #方差设置，使用已有变量

​        is_scale: true                                                                         #图像元素是否除255.，即[0,255]到[0,1]

​    \- Permute: {}                                                                             #通道变换HWC->CHW,一般都需要

  batch_size: 128                                                                              #训练时batchsize

  shuffle: true                                                                                #数据集是否shuffle

  drop_last: false                                                                             #数据集对batchsize取余数量是否丢弃



EvalReader:

  sample_transforms:                                                                           #数据预处理变换设置，意义同TrainReader

​    \- TopDownAffine:                                                                          #Affine变换设置

​        trainsize: *trainsize                                                                  #训练尺寸同上trainsize，使用已有变量

​        use_udp: true                                                                          #是否使用udp_unbias，与训练需对应

  batch_transforms:

​    \- NormalizeImage:                                                                         #图片归一化，与训练需对应

​        mean: *global_mean

​        std: *global_std

​        is_scale: true

​    \- Permute: {}                                                                             #通道变换HWC->CHW

  batch_size: 16                                                                               #测试时batchsize



TestReader:

  inputs_def:

​    image_shape: [3, *train_height, *train_width]                                              #输入数据维度设置，CHW

  sample_transforms:

​    \- Decode: {}                                                                              #图片加载

​    \- TopDownEvalAffine:                                                                      #Affine类，Eval时用

​        trainsize: *trainsize                                                                  #输入图片尺度

​    \- NormalizeImage:                                                                         #输入图像归一化

​        mean: *global_mean                                                                     #均值

​        std: *global_std                                                                       #方差

​        is_scale: true                                                                         #图像元素是否除255.，即[0,255]到[0,1]

​    \- Permute: {}                                                                             #通道变换HWC->CHW

  batch_size: 1                                                                                #Test batchsize

  fuse_normalize: false                                                                        #导出模型时是否内融合归一化操作（若是，预处理中可省略normalize，可以加快pipeline速度）
```
