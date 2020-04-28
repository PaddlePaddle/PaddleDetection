# 模型超参数配置教程

标签（空格分隔）： 模型文档

---

## 主要可以从以下几方面调整参数？
* 数据增强相关参数
* bbox及其taregt生成、匹配、采样相关超参数
* 训练相关的参数

## 参数说明
### 数据增强
| 参数        |  说明  |
| --------    | -----  |
| DecodeImage          | 读取图片为numpy数组 |
| MultiscaleTestResize | 以max_size为上限，将图像缩放到目标大小中的每个大小 |
| ResizeImage          | 以max_size为上限，将图像缩放到指定的目标大小 |
| RandomFlipImage      | 随机翻转图片 |
| NormalizeImage       | 归一化图片 |
| RandomDistort        | 随机调整亮度、对比度、饱和度 |
| ExpandImage          | 扩展图片及其中的bbox | 
| CropImage            | 裁剪图片及其中bbox | 
| NormalizeBox         | 将bbox的坐标归一化到0-1之间 | 
| Permute              | 变换数据通道顺序 |
| MixupImage           | 混合多张图片为一张，包括其中的bbox|
| RandomInterpImage    | 随机调用不同的插值方法来调整图片尺寸 | 
| ColorDistort         | 随机调整颜色值 | 
| PadBox               | 通过补0方式，将bbox补齐到最大数目 |

### RCNN系列模型主要参数设置
#### anchor_generator
| 参数        |  说明  |
| --------    | -----  |
| anchor_sizes  | 以绝对像素的形式表示anchor大小，例如[64.,128.,256.,512.]。若anchor的大小为64，则这个anchor的面积等于64**2 |
| aspect_ratios | anchor的高宽比，例如[0.5,1.0,2.0] |
| stride        | anchor在宽度和高度方向上的步长，例如[16.0,16.0] |


#### rpn_target_assign
| 参数        |  说明  |
| --------    | -----  |
| rpn_batch_size_per_im | 每张图像中RPN Sample总数 |
| rpn_fg_fraction       | 标记为foreground boxes的数量占batch内总体boxes的比例 |
| rpn_positive_overlap  | 和任意一个groundtruth box的IoU超出了阈值 rpn_positive_overlap的box被判定为正类别 |
| rpn_straddle_thresh   | 超出图像外部straddle_thresh个像素的anchors会被删除 |

#### train_proposal & test_proposal
训练和预测阶段，RPN生成Proposal的相关参数设置
| 参数        |  说明  |
| --------    | -----  |
| min_size       | 根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉 |
| nms_thresh     | NMS中的阈值 |
| pre_nms_top_n  | 每张图在NMS操作之前要保留的总框数 |
| post_nms_top_n | 每个图在NMS后要保留的总框数 |

#### BBoxAssigner
| 参数        |  说明  |
| --------    | -----  |
| batch_size_per_im | 每张图片抽取出的的RoIs的数目 |
| bbox_reg_weights  | Box 回归权重 |
| bg_thresh_hi      | background重叠阀值的上界，用于筛选background boxes |
| bg_thresh_lo      | background重叠阀值的下界，用于筛选background boxes |
| fg_fraction       | 在单张图片中，foreground boxes占所有boxes的比例 |
| fg_thresh         | foreground重叠阀值，用于筛选foreground boxes |

#### MaskAssigner
| 参数        |  说明  |
| --------    | -----  |
| resolution | 特征图分辨率大小 |

### YOLO系列模型主要参数设置
#### YOLOv3Head
| 参数        |  说明    |
| --------    | -------- |
| anchor_masks           | 当前YOLOv3损失计算中使用anchor的mask索引 |
| anchors                | YOLOv3预选框大小以及对应的数量 |
| norm_decay             | 归一化层权值的权值衰减 |

#### YOLOv3Head NMS
| 参数        |  说明    |
| --------    | -------- |
| background_label   | 背景标签（类别）的索引，如果设置为 0 ，则忽略背景标签（类别），如果设置为 -1 ，则考虑所有类别 |
| keep_top_k         | NMS步骤后每个图像要保留的总bbox数， -1表示在NMS步骤之后保留所有bbox |
| nms_threshold      | 在NMS中用于剔除检测框IOU的阈值 |
| nms_top_k          | 基于 score_threshold 的过滤检测后，根据置信度保留的最大检测次数 |
| normalized         | 检测是否已经经过正则化 |
| score_threshold    | 过滤掉低置信度分数的边界框的阈值，如果没有提供，请考虑所有边界框 |

#### YOLOv3Loss
| 参数        |  说明  |
| --------    | -----  |
| ignore_thresh | 一定条件下忽略某框置信度损失的忽略阈值 |
| label_smooth  | 在计算分类损失时将平滑分类目标，将正样本的目标平滑到1.0-1.0 / class_num，并将负样本的目标平滑到1.0 / class_num |

### SSD系列模型主要参数设置
#### output_decoder
| 参数        |  说明  |
| --------    | -----  |
| background_label | 背景标签类别值，背景标签类别上不做NMS，若设为-1，将考虑所有类别 |
| keep_top_k       | NMS操作后，要挑选的bounding box总数 |
| nms_eta          | 一种adaptive NMS的参数，仅当该值小于1.0时才起作用 |
| nms_threshold    | 用于NMS的阈值 |
| nms_top_k        |  基于score_threshold过滤预测框后，NMS操作前，要挑选出的置信度高的预测框的个数 |
| score_threshold  | 置信度得分阈值（Threshold），在NMS之前用来过滤低置信数的边界框（bounding box）。若未提供，则考虑所有框 |

#### MultiBoxHead
| 参数        |  说明  |
| --------    | -----  |
| aspect_ratios | 候选框的宽高比， aspect_ratios 和 input 的个数必须相等。如果每个特征层提取先验框的 aspect_ratio 多余一个，写成嵌套的list，例如[[2., 3.]] |
| min_ratio     | 先验框的长度和 base_size 的最小比率，注意，这里是百分比，假如比率为0.2，这里应该给20.0 |
| max_ratio     | 先验框的长度和 base_size 的最大比率，注意事项同 min_ratio  |
| min_sizes     | 每层提取的先验框的最小长度，如果输入个数len(inputs)<= 2，则必须设置 min_sizes ，并且 min_sizes 的个数应等于len(inputs) |
| max_sizes     | 每层提取的先验框的最大长度，如果len(inputs）<= 2，则必须设置 max_sizes ，并且 min_sizes 的长度应等于len(inputs) |
| steps         | 相邻先验框的中心点步长 ，如果在水平和垂直方向上步长相同，则设置steps即可，否则分别通过step_w和step_h设置不同方向的步长。如果 steps, ste_w 和 step_h 均为None，步长为输入图片的大小 base_size 和特征图大小的比例 |
| offset        | 左上角先验框中心在水平和垂直方向上的偏移 |
| flip          | 是否翻转宽高比 |
| kernel_size   | 计算回归位置和分类置信度的卷积核的大小 |
| pad           | 计算回归位置和分类置信度的卷积核的填充 |

### Retinanet系列模型主要参数设置
#### RetinaHead
| 参数        |  说明  |
| --------    | -----  |
| num_convs_per_octave  | 特征图分辨率大小 |
| num_chan              | 特征图分辨率大小 |
| max_level             | FPN的最高level |
| min_level             | FPN的最低level |
| base_scale            | anchor的基本尺寸 |
| num_scales_per_octave | 每个阶段anchor尺度的数量 |
| gamma                 | 用于平衡易分样本和难分样本的超参数， 默认值设置为2.0 |
| alpha                 | 用于平衡正样本和负样本的超参数，默认值设置为0.25 |
| sigma                 | smooth L1 loss layer的超参数 |

#### RetinaHead anchor_generator
| 参数        |  说明  |
| --------    | -----  |
| aspect_ratios | 生成anchor的高宽比，例如[0.5,1.0,2.0] |
| variance      | 在框回归delta中使用，数据类型为float32， 默认值为[0.1,0.1,0.2,0.2] |

#### RetinaHead target_assign
| 参数        |  说明  |
| --------    | -----  |
| positive_overlap | 判定anchor是一个正样本时anchor和真值框之间的最小IoU |
| negative_overlap |  判定anchor是一个负样本时anchor和真值框之间的最大IoU，默认值为0.4。该参数的设定值应小于等于positive_overlap的设定值，若大于，则positive_overlap的取值为negative_overlap的设定值 |

#### RetinaHead output_decoder
| 参数        |  说明  |
| --------    | -----  |
| score_thresh      | 在NMS步骤之前，用于滤除每个FPN层的检测框的阈值 |
| nms_thresh        | NMS步骤中用于剔除检测框的Intersection-over-Union（IoU）阈值 |
| pre_nms_top_n     | 在NMS步骤之前，保留每个FPN层的检测框的数量，默认值为1000 |
| detections_per_im | 在NMS步骤之后，每张图像要保留的检测框数量，默认值为100，若设为-1，则表示保留NMS步骤后剩下的全部检测框 |
| nms_eta           | NMS步骤中用于调整nms_threshold的参数。默认值为1.，表示nms_threshold的取值在NMS步骤中一直保持不变，即其设定值。若nms_eta小于1.，则表示当nms_threshold的取值大于0.5时，每保留一个检测框就调整一次nms_threshold的取值，即nms_threshold = nms_threshold * nms_eta，直到nms_threshold的取值小于等于0.5后结束调整 |

### 训练相关
#### LearningRate
| 参数        |  说明   |
| --------    | -----   |
| base_lr    | 用于更新参数的学习率，可以是浮点值，也可以是具有一个浮点值作为数据元素的变量 |

#### LearningRate Scheduler
| 参数        |  说明   |
| --------    | -----   |
| PiecewiseDecay | 对初始学习率进行分段衰减 |
| LinearWarmup   | 线性学习率热身(warm up)对学习率进行初步调整 |


#### optimizer
| 参数        |  说明   |
| --------    | -----   |
| momentum | Momentum优化器的动量因子 |
| type     | 优化器类型，可以SGD，Adam等 |

#### optimizer regularizer
| 参数        |  说明   |
| --------    | -----   |
| factor   | 正则化系数 |
| type     | 正则化函数类型，例如fluid.regularizer.L2DecayRegularizer |







