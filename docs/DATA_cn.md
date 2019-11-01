# 数据模块

## 介绍
本模块是一个Python模块，用于加载数据并将其转换成适用于检测模型的训练、验证、测试所需要的格式——由多个np.ndarray组成的tuple数组，例如用于Faster R-CNN模型的训练数据格式为：`[(im, im_info, im_id, gt_bbox, gt_class, is_crowd), (...)]`。

### 实现
该模块内部可分为4个子功能：数据解析、图片预处理、数据转换和数据获取接口。

我们采用`data.Dataset`表示一份数据，比如`COCO`数据包含3份数据，分别用于训练、验证和测试。原始数据存储与文件中，通过`data.source`加载到内存，然后使用`data.transform`对数据进行处理转换，最终通过`data.Reader`的接口可以获得用于训练、验证和测试的batch数据。

子功能介绍：

1. 数据解析  

数据解析得到的是`data.Dataset`,实现逻辑位于`data.source`中。通过它可以实现解析不同格式的数据集，已支持的数据源包括：

- COCO数据源

该数据集目前分为COCO2014和COCO2017，主要由json文件和image文件组成，其组织结构如下所示：

  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   |   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  │   |   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  │   |   ...
  |   ...
  ```


- Pascal VOC数据源

该数据集目前分为VOC2007和VOC2012，主要由xml文件和image文件组成，其组织结构如下所示：

  ```
  dataset/voc/
  ├── train.txt
  ├── val.txt
  ├── test.txt
  ├── label_list.txt (optional)
  ├── VOCdevkit/VOC2007
  │   ├── Annotations
  │       ├── 001789.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 001789.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  ├── VOCdevkit/VOC2012
  │   ├── Annotations
  │       ├── 003876.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 003876.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  |   ...
  ```

**说明：** 如果你在yaml配置文件中设置`use_default_label=False`, 将从`label_list.txt`
中读取类别列表，反之则可以没有`label_list.txt`文件，检测库会使用Pascal VOC数据集的默
认类别列表，默认类别列表定义在[voc\_loader.py](../ppdet/data/source/voc_loader.py)

- Roidb数据源
    该数据集主要由COCO数据集和Pascal VOC数据集转换而成的pickle文件，包含一个dict，而dict中只包含一个命名为‘records’的list（可能还有一个命名为‘cname2cid’的字典），其内容如下所示：
```python
(records, catname2clsid)
'records'是一个list并且它的结构如下:
{
    'im_file': im_fname, # 图像文件名
    'im_id': im_id, # 图像id
    'h': im_h, # 图像高度
    'w': im_w, # 图像宽度
    'is_crowd': is_crowd, # 是否重叠
    'gt_class': gt_class, # 真实框类别
    'gt_bbox': gt_bbox, # 真实框坐标
    'gt_poly': gt_poly, # 多边形坐标
}
'cname2id'是一个dict，保存了类别名到id的映射

```
我们在`./tools/`中提供了一个生成roidb数据集的代码，可以通过下面命令实现该功能。
```
# --type: 原始数据集的类别（只能是xml或者json）
# --annotation: 一个包含所需标注文件名的文件的路径
# --save-dir: 保存路径
# --samples: sample的个数（默认是-1，代表使用所有sample）
python ./ppdet/data/tools/generate_data_for_training.py
            --type=json \
            --annotation=./annotations/instances_val2017.json \
            --save-dir=./roidb \
            --samples=-1
```
 2. 图片预处理  
    图片预处理通过包括图片解码、缩放、裁剪等操作，我们采用`data.transform.operator`算子的方式来统一实现，这样能方便扩展。此外，多个算子还可以组合形成复杂的处理流程, 并被`data.transformer`中的转换器使用，比如多线程完成一个复杂的预处理流程。

 3. 数据转换器  
    数据转换器的功能是完成对某个`data.Dataset`进行转换处理，从而得到一个新的`data.Dataset`。我们采用装饰器模式实现各种不同的`data.transform.transformer`。比如用于多进程预处理的`dataset.transform.paralle_map`转换器。

 4. 数据获取接口  
     为方便训练时的数据获取，我们将多个`data.Dataset`组合在一起构成一个`data.Reader`为用户提供数据，用户只需要调用`Reader.[train|eval|infer]`即可获得对应的数据流。`Reader`支持yaml文件配置数据地址、预处理过程、加速方式等。

### APIs

主要的APIs如下：


1. 数据解析  

 - `source/coco_loader.py`：用于解析COCO数据集。[详见代码](../ppdet/data/source/coco_loader.py)
 - `source/voc_loader.py`：用于解析Pascal VOC数据集。[详见代码](../ppdet/data/source/voc_loader.py)  
 [注意]在使用VOC数据集时，若不使用默认的label列表，则需要先使用`tools/generate_data_for_training.py`生成`label_list.txt`（使用方式与数据解析中的roidb数据集获取过程一致），或提供`label_list.txt`放置于`data/pascalvoc/ImageSets/Main`中；同时在配置文件中设置参数`use_default_label`为`true`。
 - `source/loader.py`：用于解析Roidb数据集。[详见代码](../ppdet/data/source/loader.py)

2. 算子  
 `transform/operators.py`：包含多种数据增强方式，主要包括：  

```  python
RandomFlipImage：水平翻转。
RandomDistort：随机扰动图片亮度、对比度、饱和度和色相。
ResizeImage：根据特定的插值方式调整图像大小。
RandomInterpImage：使用随机的插值方式调整图像大小。
CropImage：根据缩放比例、长宽比例两个参数生成若干候选框，再依据这些候选框和标注框的面积交并比(IoU)挑选出符合要求的裁剪结果。
ExpandImage：将原始图片放进一张使用像素均值填充(随后会在减均值操作中减掉)的扩张图中，再对此图进行裁剪、缩放和翻转。
DecodeImage：以RGB格式读取图像。
Permute：对图像的通道进行排列并转为BGR格式。
NormalizeImage：对图像像素值进行归一化。
NormalizeBox：对bounding box进行归一化。
MixupImage：按比例叠加两张图像。
```
[注意]：Mixup的操作可参考[论文](https://arxiv.org/pdf/1710.09412.pdf)。

`transform/arrange_sample.py`：实现对输入网络数据的排序。  
3. 转换  
`transform/post_map.py`：用于完成批数据的预处理操作，其主要包括：

```  python
随机调整批数据的图像大小
多尺度调整图像大小
padding操作
```
`transform/transformer.py`：用于过滤无用的数据，并返回批数据。
`transform/parallel_map.py`：用于实现加速。  
4. 读取  
`reader.py`：用于组合source和transformer操作，根据`max_iter`返回batch数据。
`data_feed.py`: 用于配置 `reader.py`中所需的默认参数.




### 使用
#### 常规使用
结合yaml文件中的配置信息，完成本模块的功能。yaml文件的使用可以参见配置文件部分。

 - 读取用于训练的数据

``` python
ccfg = load_cfg('./config.yml')
coco = Reader(ccfg.DATA, ccfg.TRANSFORM, maxiter=-1)
```
#### 如何使用自定义数据集？

- 选择1：将数据集转换为COCO格式。
```
 # 在./tools/中提供了x2coco.py用于将labelme标注的数据集或cityscape数据集转换为COCO数据集
 python ./ppdet/data/tools/x2coco.py --dataset_type labelme
                                --json_input_dir ./labelme_annos/
                                --image_input_dir ./labelme_imgs/
                                --output_dir ./cocome/
                                --train_proportion 0.8
                                --val_proportion 0.2
                                --test_proportion 0.0
 # --dataset_type：需要转换的数据格式，目前支持：’labelme‘和’cityscape‘
 # --json_input_dir：使用labelme标注的json文件所在文件夹
 # --image_input_dir：图像文件所在文件夹
 # --output_dir：转换后的COCO格式数据集存放位置
 # --train_proportion：标注数据中用于train的比例
 # --val_proportion：标注数据中用于validation的比例
 # --test_proportion: 标注数据中用于infer的比例
```
- 选择2：

1. 仿照`./source/coco_loader.py`和`./source/voc_loader.py`，添加`./source/XX_loader.py`并实现`load`函数。  
2. 在`./source/loader.py`的`load`函数中添加使用`./source/XX_loader.py`的入口。  
3. 修改`./source/__init__.py`：  


```python
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
    source_type = 'RoiDbSource'
# 将上述代码替换为如下代码：
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource', 'XXSource']:
    source_type = 'RoiDbSource'
```

4. 在配置文件中修改`dataset`下的`type`为`XXSource`。  

#### 如何增加数据预处理？
- 若增加单张图像的增强预处理，可在`transform/operators.py`中参考每个类的代码，新建一个类来实现新的数据增强；同时在配置文件中增加该预处理。
- 若增加单个batch的图像预处理，可在`transform/post_map.py`中参考`build_post_map`中每个函数的代码，新建一个内部函数来实现新的批数据预处理；同时在配置文件中增加该预处理。
