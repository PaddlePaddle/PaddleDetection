# 数据处理模块

## 目录
- [1.简介](#1.简介)
- [2.数据集](#2.数据集)
  - [2.1COCO数据集](#2.1COCO数据集)
  - [2.2Pascal VOC数据集](#2.2Pascal-VOC数据集)
  - [2.3自定义数据集](#2.3自定义数据集)
- [3.数据预处理](#3.数据预处理)
  - [3.1数据增强算子](#3.1数据增强算子)
  - [3.2自定义数据增强算子](#3.2自定义数据增强算子)
- [4.Raeder](#4.Reader)
- [5.配置及运行](#5.配置及运行)
  - [5.1配置](#5.1配置)
  - [5.2运行](#5.2运行)

### 1.简介
PaddleDetection的数据处理模块的所有代码逻辑在`ppdet/data/`中，数据处理模块用于加载数据并将其转换成适用于物体检测模型的训练、评估、推理所需要的格式。
数据处理模块的主要构成如下架构所示：
```bash
  ppdet/data/
  ├── reader.py     # 基于Dataloader封装的Reader模块
  ├── source  # 数据源管理模块
  │   ├── dataset.py      # 定义数据源基类，各类数据集继承于此
  │   ├── coco.py         # COCO数据集解析与格式化数据
  │   ├── voc.py          # Pascal VOC数据集解析与格式化数据
  │   ├── widerface.py    # WIDER-FACE数据集解析与格式化数据
  │   ├── category.py    # 相关数据集的类别信息
  ├── transform  # 数据预处理模块
  │   ├── batch_operators.py  # 定义各类基于批量数据的预处理算子
  │   ├── op_helper.py    # 预处理算子的辅助函数
  │   ├── operators.py    # 定义各类基于单张图片的预处理算子
  │   ├── gridmask_utils.py    # GridMask数据增强函数
  │   ├── autoaugment_utils.py  # AutoAugment辅助函数
  ├── shm_utils.py     # 用于使用共享内存的辅助函数
  ```


### 2.数据集
数据集定义在`source`目录下，其中`dataset.py`中定义了数据集的基类`DetDataSet`, 所有的数据集均继承于基类，`DetDataset`基类里定义了如下等方法：

| 方法                        | 输入   | 输出           |  备注                   |
| :------------------------: | :----: | :------------: | :--------------: |
| \_\_len\_\_ | 无     | int, 数据集中样本的数量     | 过滤掉了无标注的样本 |
| \_\_getitem\_\_ | int, 样本的索引idx     |  dict, 索引idx对应的样本roidb  | 得到transform之后的样本roidb |
| check_or_download_dataset            | 无     | 无  |  检查数据集是否存在，如果不存在则下载，目前支持COCO, VOC，widerface等数据集 |
| set_kwargs                |  可选参数，以键值对的形式给出   | 无  | 目前用于支持接收mixup, cutmix等参数的设置 |
| set_transform            | 一系列的transform函数   | 无  | 设置数据集的transform函数 |
| set_epoch            | int, 当前的epoch  | 无  | 用于dataset与训练过程的交互 |
| parse_dataset            | 无  | 无  | 用于从数据中读取所有的样本 |
| get_anno            | 无  | 无  | 用于获取标注文件的路径 |

当一个数据集类继承自`DetDataSet`，那么它只需要实现parse_dataset函数即可。parse_dataset根据数据集设置的数据集根路径dataset_dir，图片文件夹image_dir， 标注文件路径anno_path取出所有的样本，并将其保存在一个列表roidbs中，每一个列表中的元素为一个样本xxx_rec(比如coco_rec或者voc_rec)，用dict表示，dict中包含样本的image, gt_bbox, gt_class等字段。COCO和Pascal-VOC数据集中的xxx_rec的数据结构定义如下：
  ```python
  xxx_rec = {
      'im_file': im_fname,         # 一张图像的完整路径
      'im_id': np.array([img_id]), # 一张图像的ID序号
      'h': im_h,                   # 图像高度
      'w': im_w,                   # 图像宽度
      'is_crowd': is_crowd,        # 是否是群落对象, 默认为0 (VOC中无此字段)
      'gt_class': gt_class,        # 标注框标签名称的ID序号
      'gt_bbox': gt_bbox,          # 标注框坐标(xmin, ymin, xmax, ymax)
      'gt_poly': gt_poly,          # 分割掩码，此字段只在coco_rec中出现，默认为None
      'difficult': difficult       # 是否是困难样本，此字段只在voc_rec中出现，默认为0
  }
  ```

xxx_rec中的内容也可以通过`DetDataSet`的data_fields参数来控制，即可以过滤掉一些不需要的字段，但大多数情况下不需要修改，按照`configs/datasets`中的默认配置即可。

此外，在parse_dataset函数中，保存了类别名到id的映射的一个字典`cname2cid`。在coco数据集中，会利用[COCO API](https://github.com/cocodataset/cocoapi)从标注文件中加载数据集的类别名，并设置此字典。在voc数据集中，如果设置`use_default_label=False`，将从`label_list.txt`中读取类别列表，反之将使用voc默认的类别列表。

#### 2.1COCO数据集
COCO数据集目前分为COCO2014和COCO2017，主要由json文件和image文件组成，其组织结构如下所示：

  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   │   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  │   │   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  │   │   ...
  ```

在`source/coco.py`中定义并注册了`COCODataSet`数据集类，其继承自`DetDataSet`，并实现了parse_dataset方法，调用[COCO API](https://github.com/cocodataset/cocoapi)加载并解析COCO格式数据源`roidbs`和`cname2cid`，具体可参见`source/coco.py`源码。将其他数据集转换成COCO格式可以参考[用户数据转成COCO数据](../tutorials/data/PrepareDetDataSet.md#用户数据转成COCO数据)

#### 2.2Pascal VOC数据集
该数据集目前分为VOC2007和VOC2012，主要由xml文件和image文件组成，其组织结构如下所示：
```
  dataset/voc/
  ├── trainval.txt
  ├── test.txt
  ├── label_list.txt (optional)
  ├── VOCdevkit/VOC2007
  │   ├── Annotations
  │       ├── 001789.xml
  │       │   ...
  │   ├── JPEGImages
  │       ├── 001789.jpg
  │       │   ...
  │   ├── ImageSets
  │       |   ...
  ├── VOCdevkit/VOC2012
  │   ├── Annotations
  │       ├── 2011_003876.xml
  │       │   ...
  │   ├── JPEGImages
  │       ├── 2011_003876.jpg
  │       │   ...
  │   ├── ImageSets
  │       │   ...
  ```
在`source/voc.py`中定义并注册了`VOCDataSet`数据集，它继承自`DetDataSet`基类，并重写了`parse_dataset`方法，解析VOC数据集中xml格式标注文件，更新`roidbs`和`cname2cid`。将其他数据集转换成VOC格式可以参考[用户数据转成VOC数据](../tutorials/data/PrepareDetDataSet.md#用户数据转成VOC数据)

#### 2.3自定义数据集
如果COCODataSet和VOCDataSet不能满足你的需求，可以通过自定义数据集的方式来加载你的数据集。只需要以下两步即可实现自定义数据集

1. 新建`source/xxx.py`，定义类`XXXDataSet`继承自`DetDataSet`基类，完成注册与序列化，并重写`parse_dataset`方法对`roidbs`与`cname2cid`更新：
  ```python
  from ppdet.core.workspace import register, serializable

  #注册并序列化
  @register
  @serializable
  class XXXDataSet(DetDataSet):
      def __init__(self,
                  dataset_dir=None,
                  image_dir=None,
                  anno_path=None,
                  ...
                  ):
          self.roidbs = None
          self.cname2cid = None
          ...

      def parse_dataset(self):
          ...
          省略具体解析数据逻辑
          ...
          self.roidbs, self.cname2cid = records, cname2cid
  ```

2. 在`source/__init__.py`中添加引用：
  ```python
  from . import xxx
  from .xxx import *
  ```
完成以上两步就将新的数据源`XXXDataSet`添加好了，你可以参考[配置及运行](#5.配置及运行)实现自定义数据集的使用。

### 3.数据预处理

#### 3.1数据增强算子
PaddleDetection中支持了种类丰富的数据增强算子，有单图像数据增强算子与批数据增强算子两种方式，您可选取合适的算子组合使用。单图像数据增强算子定义在`transform/operators.py`中，已支持的单图像数据增强算子详见下表：

| 名称                     |  作用                   |
| :---------------------: | :--------------: |
| Decode             | 从图像文件或内存buffer中加载图像，格式为RGB格式 |
| Permute                 | 假如输入是HWC顺序变成CHW |
| RandomErasingImage | 对图像进行随机擦除 |
| NormalizeImage          | 对图像像素值进行归一化，如果设置is_scale=True，则先将像素值除以255.0, 再进行归一化。 |
| GridMask  | GridMask数据增广 |
| RandomDistort           | 随机扰动图片亮度、对比度、饱和度和色相 |
| AutoAugment | AutoAugment数据增广，包含一系列数据增强方法 |
| RandomFlip         | 随机水平翻转图像 |
| Resize             | 对于图像进行resize，并对标注进行相应的变换 |
| MultiscaleTestResize    | 将图像重新缩放为多尺度list的每个尺寸 |
| RandomResize | 对于图像进行随机Resize，可以Resize到不同的尺寸以及使用不同的插值策略 |
| RandomExpand | 将原始图片放入用像素均值填充的扩张图中，对此图进行裁剪、缩放和翻转 |
| CropWithSampling         | 根据缩放比例、长宽比例生成若干候选框，再依据这些候选框和标注框的面积交并比(IoU)挑选出符合要求的裁剪结果 |
| CropImageWithDataAchorSampling | 基于CropImage，在人脸检测中，随机将图片尺度变换到一定范围的尺度，大大增强人脸的尺度变化 |
| RandomCrop              | 原理同CropImage，以随机比例与IoU阈值进行处理 |
| RandomScaledCrop        | 根据长边对图像进行随机裁剪，并对标注做相应的变换 |
| Cutmix             | Cutmix数据增强，对两张图片做拼接  |
| Mixup              | Mixup数据增强，按比例叠加两张图像 |
| NormalizeBox            | 对bounding box进行归一化 |
| PadBox                  | 如果bounding box的数量少于num_max_boxes，则将零填充到bbox |
| BboxXYXY2XYWH           | 将bounding box从(xmin,ymin,xmax,ymin)形式转换为(xmin,ymin,width,height)格式 |
| Pad           | 将图片Pad某一个数的整数倍或者指定的size，并支持指定Pad的方式 |
| Poly2Mask | Poly2Mask数据增强 ｜

批数据增强算子定义在`transform/batch_operators.py`中, 目前支持的算子列表如下：
| 名称                     |  作用                   |
| :---------------------: | :--------------: |
| PadBatch           | 随机对每个batch的数据图片进行Pad操作，使得batch中的图片具有相同的shape |
| BatchRandomResize  | 对一个batch的图片进行resize，使得batch中的图片随机缩放到相同的尺寸  |
| Gt2YoloTarget      | 通过gt数据生成YOLO系列模型的目标  |
| Gt2FCOSTarget      | 通过gt数据生成FCOS模型的目标 |
| Gt2TTFTarget       | 通过gt数据生成TTFNet模型的目标 |
| Gt2Solov2Target    | 通过gt数据生成SOLOv2模型的目标 |

**几点说明：**
- 数据增强算子的输入为sample或者samples，每一个sample对应上文所说的`DetDataSet`输出的roidbs中的一个样本，如coco_rec或者voc_rec
- 单图像数据增强算子(Mixup, Cutmix等除外)也可用于批数据处理中。但是，单图像处理算子和批图像处理算子仍有一些差异，以RandomResize和BatchRandomResize为例，RandomResize会将一个Batch中的每张图片进行随机缩放，但是每一张图像Resize之后的形状不尽相同，BatchRandomResize则会将一个Batch中的所有图片随机缩放到相同的形状。
- 除BatchRandomResize外，定义在`transform/batch_operators.py`的批数据增强算子接收的输入图像均为CHW形式，所以使用这些批数据增强算子前请先使用Permute进行处理。如果用到Gt2xxxTarget算子，需要将其放置在靠后的位置。NormalizeBox算子建议放置在Gt2xxxTarget之前。将这些限制条件总结下来，推荐的预处理算子的顺序为
  ```
    - XXX: {}
    - ...
    - BatchRandomResize: {...} # 如果不需要，可以移除，如果需要，放置在Permute之前
    - Permute: {} # 必须项
    - NormalizeBox: {} # 如果需要，建议放在Gt2XXXTarget之前
    - PadBatch: {...} # 如果不需要可移除，如果需要，建议放置在Permute之后
    - Gt2XXXTarget: {...} # 建议与PadBatch放置在最后的位置
  ```

#### 3.2自定义数据增强算子
如果需要自定义数据增强算子，那么您需要了解下数据增强算子的相关逻辑。数据增强算子基类为定义在`transform/operators.py`中的`BaseOperator`类，单图像数据增强算子与批数据增强算子均继承自这个基类。完整定义参考源码，以下代码显示了`BaseOperator`类的关键函数: apply和__call__方法
  ``` python
  class BaseOperator(object):

    ...

    def apply(self, sample, context=None):
        return sample

    def __call__(self, sample, context=None):
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample
  ```
__call__方法为`BaseOperator`的调用入口，接收一个sample(单图像)或者多个sample(多图像)作为输入，并调用apply函数对一个或者多个sample进行处理。大多数情况下，你只需要继承`BaseOperator`重写apply方法或者重写__call__方法即可，如下所示，定义了一个XXXOp继承自BaseOperator，并注册：
  ```python
  @register_op
  class XXXOp(BaseOperator):
    def __init__(self,...):

      super(XXXImage, self).__init__()
      ...

    # 大多数情况下只需要重写apply方法
    def apply(self, sample, context=None):
      ...
      省略对输入的sample具体操作
      ...
      return sample

    # 如果有需要，可以重写__call__方法，如Mixup, Gt2XXXTarget等
    # def __call__(self, sample, context=None):
    #   ...
    #   省略对输入的sample具体操作
    #   ...
    #   return sample
  ```
大多数情况下，只需要重写apply方法即可，如`transform/operators.py`中除Mixup和Cutmix外的预处理算子。对于批处理的情况一般需要重写__call__方法，如`transform/batch_operators.py`的预处理算子。

### 4.Reader
Reader相关的类定义在`reader.py`, 其中定义了`BaseDataLoader`类。`BaseDataLoader`在`paddle.io.DataLoader`的基础上封装了一层，其具备`paddle.io.DataLoader`的所有功能，并能够实现不同模型对于`DetDataset`的不同需求，如可以通过对Reader进行设置，以控制`DetDataset`支持Mixup, Cutmix等操作。除此之外，数据预处理算子通过`Compose`类和`BatchCompose`类组合起来分别传入`DetDataset`和`paddle.io.DataLoader`中。
所有的Reader类都继承自`BaseDataLoader`类，具体可参见源码。

### 5.配置及运行

#### 5.1 配置
与数据预处理相关的模块的配置文件包含所有模型公用的Dataset的配置文件，以及不同模型专用的Reader的配置文件。

##### 5.1.1 Dataset配置
关于Dataset的配置文件存在于`configs/datasets`文件夹。比如COCO数据集的配置文件如下：
```
metric: COCO # 目前支持COCO, VOC, OID， WiderFace等评估标准
num_classes: 80 # num_classes数据集的类别数，不包含背景类

TrainDataset:
  !COCODataSet
    image_dir: train2017 # 训练集的图片所在文件相对于dataset_dir的路径
    anno_path: annotations/instances_train2017.json # 训练集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco #数据集所在路径，相对于PaddleDetection路径
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd'] # 控制dataset输出的sample所包含的字段，注意此为TrainDataset独有的且必须配置的字段

EvalDataset:
  !COCODataSet
    image_dir: val2017 # 验证集的图片所在文件夹相对于dataset_dir的路径
    anno_path: annotations/instances_val2017.json # 验证集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco # 数据集所在路径，相对于PaddleDetection路径

TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val2017.json # 标注文件所在路径，仅用于读取数据集的类别信息，支持json和txt格式
    dataset_dir: dataset/coco # 数据集所在路径，若添加了此行，则`anno_path`路径为`dataset_dir/anno_path`，若此行不设置或去掉此行，则`anno_path`路径即为`anno_path`
```
在PaddleDetection的yml配置文件中，使用`!`直接序列化模块实例(可以是函数，实例等)，上述的配置文件均使用Dataset进行了序列化。

**注意：**
请运行前自行仔细检查数据集的配置路径，在训练或验证时如果TrainDataset和EvalDataset的路径配置有误，会提示自动下载数据集。若使用自定义数据集，在推理时如果TestDataset路径配置有误，会提示使用默认COCO数据集的类别信息。


##### 5.1.2 Reader配置
不同模型专用的Reader定义在每一个模型的文件夹下，如yolov3的Reader配置文件定义在`configs/yolov3/_base_/yolov3_reader.yml`。一个Reader的示例配置如下：
```
worker_num: 2
TrainReader:
  sample_transforms:
    - Decode: {}
    ...
  batch_transforms:
    ...
  batch_size: 8
  shuffle: true
  drop_last: true
  use_shared_memory: true

EvalReader:
  sample_transforms:
    - Decode: {}
    ...
  batch_size: 1

TestReader:
  inputs_def:
    image_shape: [3, 608, 608]
  sample_transforms:
    - Decode: {}
    ...
  batch_size: 1
```
你可以在Reader中定义不同的预处理算子，每张卡的batch_size以及DataLoader的worker_num等。

#### 5.2运行
在PaddleDetection的训练、评估和测试运行程序中，都通过创建Reader迭代器。Reader在`ppdet/engine/trainer.py`中创建。下面的代码展示了如何创建训练时的Reader
``` python
from ppdet.core.workspace import create
# build data loader
self.dataset = cfg['TrainDataset']
self.loader = create('TrainReader')(selfdataset, cfg.worker_num)
```
相应的预测以及评估时的Reader与之类似，具体可参考`ppdet/engine/trainer.py`源码。

> 关于数据处理模块，如您有其他问题或建议，请给我们提issue，我们非常欢迎您的反馈。
