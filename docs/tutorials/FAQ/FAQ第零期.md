# FAQ：第零期

**Q:**  为什么我使用单GPU训练loss会出`NaN`? </br>
**A:**  配置文件中原始学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整学习率（例如，除以8）。

以[faster_rcnn_r50](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/faster_rcnn/faster_rcnn_r50_1x_coco.yml) 为例,在静态图下计算规则表如下所示，它们是等价的，表中变化节点即为`piecewise decay`里的`boundaries`: </br>


| GPU数  |batch size/卡| 学习率  | 最大轮数 | 变化节点       |
| :---------: |  :------------:|:------------: | :-------: | :--------------: |
| 2          | 1 | 0.0025         | 720000    | [480000, 640000] |
| 4          | 1 | 0.005          | 360000    | [240000, 320000] |
| 8          | 1| 0.01           | 180000    | [120000, 160000] |

* 上述方式适用于静态图下。在动态图中，由于训练以epoch方式计数，因此调整GPU卡数后只需要修改学习率即可，修改方式和静态图相同.


**Q:**  自定义数据集时，配置文件里的`num_classes`应该如何设置? </br>
**A:**  动态图中，自定义数据集时将`num_classes`统一设置为自定义数据集的类别数即可，静态图中(static目录下)，YOLO系列模型和anchor free系列模型将`num_classes`设置为自定义数据集类别即可，其他模型如RCNN系列，SSD，RetinaNet，SOLOv2等模型，由于检测原理上分类中需要区分背景框和前景框，设置的`num_classes`须为自定义数据集类别数+1，即增加一类背景类。

**Q:**  PP-YOLOv2模型训练使用`—eval`做训练中验证，在第一次做eval的时候hang住,该如何处理?</br>
**A:**  PP-YOLO系列模型如果只加载backbone的预训练权重从头开始训练的话收敛会比较慢，当模型还没有较好收敛的时候做预测时，由于输出的预测框比较混乱，在NMS时做排序和滤除会非常耗时，就好像eval时hang住了一样，这种情况一般发生在使用自定义数据集并且自定义数据集样本数较少导致训练到第一次做eval的时候训练轮数较少，模型还没有较好收敛的情况下，可以通过如下三个方面排查解决。



* PaddleDetection中提供的默认配置一般是采用8卡训练的配置，配置文件中的`batch_size`数为每卡的batch size，若训练的时候不是使用8卡或者对`batch_size`有修改，需要等比例的调小初始`learning_rate`来获得较好的收敛效果

* 如果使用自定义数据集并且样本数比较少，建议增大`snapshot_epoch`数来增加第一次进行eval的时候的训练轮数来保证模型已经较好收敛

* 若使用自定义数据集训练，可以加载我们发布的COCO或VOC数据集上训练好的权重进行finetune训练来加快收敛速度，可以使用`-o pretrain_weights=xxx`的方式指定预训练权重，xxx可以是Model Zoo里发布的模型权重链接




**Q:**  如何更好的理解reader和自定义修改reader文件
```
# 每张GPU reader进程个数
worker_num: 2
# 训练数据
TrainReader:
  inputs_def:
    num_max_boxes: 50
  # 训练数据transforms
  sample_transforms:
    - Decode: {} # 图片解码，将图片数据从numpy格式转为rgb格式，是必须存在的一个OP
    - Mixup: {alpha: 1.5, beta: 1.5} # Mixup数据增强，对两个样本的gt_bbbox/gt_score操作，构建虚拟的训练样本，可选的OP
    - RandomDistort: {} # 随机颜色失真，可选的OP
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]} # 随机Canvas填充，可选的OP
    - RandomCrop: {} # 随机裁剪，可选的OP
    - RandomFlip: {} # 随机左右翻转，默认概率0.5，可选的OP
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
  # 读取数据是是否乱序
  shuffle: true
  # 是否丢弃最后不能完整组成batch的数据
  drop_last: true
  # mixup_epoch，大于最大epoch，表示训练过程一直使用mixup数据增广。默认值为-1，表示不使用Mixup。如果删去- Mixup: {alpha: 1.5, beta: 1.5}这行代码则必须也将mixup_epoch设置为-1或者删除
  mixup_epoch: 25000
  # 是否通过共享内存进行数据读取加速，需要保证共享内存大小(如/dev/shm)满足大于1G
  use_shared_memory: true

  如果需要单尺度训练，则去掉batch_transforms里的BatchRandomResize这一行，在sample_transforms最后一行添加- Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}

  Decode是必须保留的，如果想要去除数据增强，则可以注释或删除Mixup RandomDistort RandomExpand RandomCrop RandomFlip，注意如果注释或删除Mixup则必须也将mixup_epoch这一行注释或删除，或者设置为-1表示不使用Mixup
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [608, 608], keep_ratio: False, interp: 2}

```
**Q:**  用户如何控制类别类别输出？即图中有多类目标只输出其中的某几类

**A:**  用户可自行在代码中进行修改，增加条件设置。
```
# filter by class_id
keep_class_id = [1, 2]
bbox_res = [e for e in bbox_res if int(e[0]) in keep_class_id]
```
https://github.com/PaddlePaddle/PaddleDetection/blob/b87a1ea86fa18ce69e44a17ad1b49c1326f19ff9/ppdet/engine/trainer.py#L438

**Q:**  用户自定义数据集训练，预测结果标签错误

**A:**  此类情况往往是用户在设置数据集路径时候，并没有关注TestDataset中anno_path的路径问题。需要用户将anno_path设置成自己的路径。
```
TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val2017.json
```

**Q:** 如何打印网络FLOPs？

**A:** 在`configs/runtime.yml`中设置`print_flops: true`，同时需要安装PaddleSlim(比如：pip install paddleslim)，即可打印模型的FLOPs。

**Q:** 如何使用无标注框进行训练？

**A:** 在`configs/dataset/coco.py` 或者`configs/dataset/voc.py`中的TrainDataset下设置`allow_empty: true`, 此时允许数据集加载无标注框进行训练。该功能支持coco，voc数据格式，RCNN系列和YOLO系列模型验证能够正常训练。另外，如果无标注框数据过多，会影响模型收敛，在TrainDataset下可以设置`empty_ratio: 0.1`对无标注框数据进行随机采样，控制无标注框的数据量占总数据量的比例，默认值为1.，即使用全部无标注框
